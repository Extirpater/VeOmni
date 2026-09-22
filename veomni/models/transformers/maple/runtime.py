"""Maple I-DLM masks, strided decoding, and runtime helpers.

Training follows Appendix E of arXiv:2604.11035: noisy queries see causal
positions in their own block and clean preceding blocks; clean queries see
only their causal clean prefix. A triangular mask alone is insufficient.
"""

import math
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from veomni.arguments import AcceleratorConfig, OpsImplementationConfig
from veomni.distributed.parallel_state import clear_parallel_state, init_parallel_state_from_config
from veomni.models.auto import build_foundation_model
from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device


def shifted_targets(labels, position_ids):
    targets = F.pad(labels[..., 1:], (0, 1), value=-100)
    targets[..., :-1].masked_fill_(position_ids[..., 1:] <= position_ids[..., :-1], -100)
    return targets


def prepare_idlm_inputs(input_ids, labels, position_ids, mask_token_id):
    """Keep prompt tokens visible, fully mask supervised tokens, and shift once."""
    targets = shifted_targets(labels, position_ids)
    noisy_targets = targets.masked_fill(labels == -100, -100)
    noisy = input_ids.masked_fill(labels != -100, mask_token_id)
    return torch.cat((noisy, input_ids), -1), torch.cat((position_ids, position_ids), -1), noisy_targets, targets


def idlm_mask_mod(position_ids, valid_tokens, block_size, sliding_window=None):
    if block_size < 1:
        raise ValueError("block_size is the number of masks per block and must be positive")
    length = position_ids.shape[-1]
    segments = (position_ids == 0).to(torch.int32).cumsum(-1)

    def mask_mod(batch, head, query, key):
        q, k = query % length, key % length
        qp, kp = position_ids[batch, q], position_ids[batch, k]
        q_clean, k_clean = query >= length, key >= length
        noisy_self = ~q_clean & ~k_clean & (qp // block_size == kp // block_size) & (kp <= qp)
        noisy_cross = ~q_clean & k_clean & (kp // block_size < qp // block_size)
        clean_self = q_clean & k_clean & (kp <= qp)
        visible = noisy_self | noisy_cross | clean_self
        visible = visible & (segments[batch, q] == segments[batch, k])
        visible = visible & valid_tokens[batch, q] & valid_tokens[batch, k]
        visible = visible & (query < 2 * length) & (key < 2 * length)
        if sliding_window is not None:
            # Maple's reference FA wrapper uses window_size=(sliding_window, 0).
            visible = visible & (qp - kp <= sliding_window)
        return visible

    return mask_mod


def _idlm_tile_masks(position_ids, valid_tokens, block_size, sliding_window, tile_size=128):
    """Bound visibility per tile without constructing a token-by-token mask.

    Partial tiles may conservatively contain invisible pairs; mask_mod checks
    those pairs in attention. A full tile must prove that every pair is visible.
    Bounds also cover tiles crossing documents or the noisy/clean boundary.
    """
    batch, length = position_ids.shape
    tiles = (2 * length + tile_size - 1) // tile_size
    indices = torch.arange(tiles * tile_size, device=position_ids.device)
    source = indices % length
    positions = position_ids[:, source].reshape(batch, tiles, tile_size)
    segments = (position_ids == 0).to(torch.int32).cumsum(-1)[:, source].reshape(batch, tiles, tile_size)
    valid = (valid_tokens[:, source] & (indices < 2 * length)).reshape(batch, tiles, tile_size)
    clean = (indices >= length).reshape(1, tiles, tile_size)

    pmin, pmax = positions.amin(-1), positions.amax(-1)
    smin, smax = segments.amin(-1), segments.amax(-1)
    bmin, bmax = pmin // block_size, pmax // block_size
    any_clean, all_clean = clean.any(-1), clean.all(-1)
    any_valid, all_valid = valid.any(-1), valid.all(-1)

    qmin, qmax, kmin, kmax = pmin[:, :, None], pmax[:, :, None], pmin[:, None, :], pmax[:, None, :]
    qbmin, qbmax, kbmin, kbmax = bmin[:, :, None], bmax[:, :, None], bmin[:, None, :], bmax[:, None, :]
    possible = (
        (~all_clean[:, :, None] & ~all_clean[:, None, :] & (qbmin <= kbmax) & (kbmin <= qbmax) & (kmin <= qmax))
        | (~all_clean[:, :, None] & any_clean[:, None, :] & (kbmin < qbmax))
        | (any_clean[:, :, None] & any_clean[:, None, :] & (kmin <= qmax))
    )
    possible &= (smin[:, :, None] <= smax[:, None, :]) & (smin[:, None, :] <= smax[:, :, None])
    possible &= any_valid[:, :, None] & any_valid[:, None, :]

    full = (
        (
            ~any_clean[:, :, None]
            & ~any_clean[:, None, :]
            & (qbmin == qbmax)
            & (qbmin == kbmin)
            & (qbmin == kbmax)
            & (kmax <= qmin)
        )
        | (~any_clean[:, :, None] & all_clean[:, None, :] & (kbmax < qbmin))
        | (all_clean[:, :, None] & all_clean[:, None, :] & (kmax <= qmin))
    )
    full &= (smin[:, :, None] == smax[:, :, None]) & (smin[:, None, :] == smax[:, None, :])
    full &= smin[:, :, None] == smin[:, None, :]
    full &= all_valid[:, :, None] & all_valid[:, None, :]
    if sliding_window is not None:
        possible &= qmin - kmax <= sliding_window
        full &= qmax - kmin <= sliding_window
    return (possible & ~full)[:, None], full[:, None]


def make_idlm_attention_mask(position_ids, valid_tokens, block_size, *, sliding_window=None, flex=False):
    mask_mod = idlm_mask_mod(position_ids, valid_tokens.bool(), block_size, sliding_window)
    batch_size, length = position_ids.shape
    if flex:
        from torch.nn.attention.flex_attention import BlockMask

        partial, full = _idlm_tile_masks(position_ids, valid_tokens.bool(), block_size, sliding_window)

        def ordered(tiles):
            counts = tiles.sum(-1, dtype=torch.int32)
            indices = tiles.to(torch.int32).argsort(dim=-1, descending=True, stable=True).to(torch.int32)
            return counts.contiguous(), indices.contiguous()

        return BlockMask.from_kv_blocks(
            *ordered(partial),
            *ordered(full),
            BLOCK_SIZE=128,
            mask_mod=mask_mod,
            seq_lengths=(2 * length, 2 * length),
        )
    batch = torch.arange(batch_size, device=position_ids.device)[:, None, None]
    query = torch.arange(2 * length, device=position_ids.device)[None, :, None]
    key = torch.arange(2 * length, device=position_ids.device)[None, None, :]
    return mask_mod(batch, 0, query, key)[:, None]


def balanced_idlm_loss(masked_loss, clean_loss, clean_weight=0.2, auto_balance=False):
    scale = masked_loss.detach() / clean_loss.detach().clamp_min(1e-8) if auto_balance else clean_weight
    return masked_loss + scale * clean_loss


@dataclass
class ISDResult:
    sequences: torch.Tensor
    proposed: int
    accepted: int
    forward_passes: int
    processed_tokens: int = 0


@torch.no_grad()
def introspective_generate(
    model,
    input_ids,
    *,
    mask_token_id,
    max_new_tokens=128,
    stride=4,
    temperature=1.0,
    eos_token_id=None,
    generator=None,
    use_cache=False,
):
    """Exact p/q sampling, fusing verification with the next masked proposal.

    With use_cache=True, retain only the accepted prefix before each forward;
    rejected tokens and mask states are rolled back. The uncached path is an
    independent reference. Both preserve the converted model's causal
    distribution. Call model.eval() first. One sequence per call.
    """
    if model.training:
        raise ValueError("ISD requires model.eval()")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError("ISD expects one nonempty prompt")
    if stride < 1 or max_new_tokens < 0 or temperature <= 0:
        raise ValueError("Invalid stride, token limit, or temperature")
    prefix = input_ids.clone()
    end_length = prefix.shape[1] + max_new_tokens
    proposed = accepted = forwards = processed_tokens = 0
    draft = None
    proposal_probs = None
    cache = None
    eos_ids = set(eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id])

    def distribution(logits):
        return (logits.float() / temperature).softmax(-1)

    def sample(probs):
        return torch.multinomial(probs, 1, generator=generator).squeeze(-1)

    def append(token):
        nonlocal prefix
        prefix = torch.cat((prefix, token.reshape(1, 1)), -1)
        return prefix.shape[1] == end_length or token.item() in eos_ids

    def run_model(sequence, confirmed_length):
        nonlocal cache, forwards, processed_tokens
        offset = 0
        if use_cache:
            if cache is not None:
                # Recompute at least the final confirmed token, whose logits
                # predict the first proposal. Later cached tokens are either
                # rejected proposals or masks and must never become a prefix.
                offset = min(cache.get_seq_length(), confirmed_length - 1)
                remove = cache.get_seq_length() - offset
                if remove:
                    cache.crop(-remove)
            queries = sequence[:, offset:]
            output = model(input_ids=queries, use_cache=True, past_key_values=cache)
            cache = output.past_key_values
            if cache is None:
                raise ValueError("The model did not return a KV cache")
        else:
            queries = sequence
            output = model(input_ids=queries, use_cache=False)
        forwards += 1
        processed_tokens += queries.numel()
        return output.logits[0], offset

    while prefix.shape[1] < end_length:
        if draft is None:
            width = min(stride, end_length - prefix.shape[1])
            masks = prefix.new_full((1, width - 1), mask_token_id)
            logits, offset = run_model(torch.cat((prefix, masks), -1), prefix.shape[1])
            probs = distribution(logits[prefix.shape[1] - 1 - offset :])
            if append(sample(probs[0])):
                break
            proposal_probs = probs[1:]
            draft = sample(proposal_probs) if width > 1 else None
            continue

        # The prefix already contains the previous stride's guaranteed anchor.
        previous_length = prefix.shape[1]
        verify_input = torch.cat((prefix, draft[None]), -1)
        next_width = min(stride, max(0, end_length - verify_input.shape[1]))
        masks = prefix.new_full((1, max(0, next_width - 1)), mask_token_id)
        logits, offset = run_model(torch.cat((verify_input, masks), -1), previous_length)
        anchors = distribution(logits[previous_length - 1 - offset : previous_length + draft.numel() - 1 - offset])
        rejected = False
        done = False
        for index, token in enumerate(draft):
            proposed += 1
            p, q = anchors[index], proposal_probs[index]
            threshold = (p[token] / q[token].clamp_min(torch.finfo(q.dtype).tiny)).clamp(max=1)
            if torch.rand((), device=prefix.device, generator=generator) < threshold:
                accepted += 1
                done = append(token)
            else:
                residual = (p - q).clamp_min(0)
                residual_sum = residual.sum()
                if not residual_sum > 0:
                    raise RuntimeError("ISD rejection has no residual probability mass")
                done = append(sample(residual / residual_sum))
                rejected = True
            if done or rejected:
                break
        if done:
            break
        if rejected:
            draft = proposal_probs = None
            continue
        next_probs = distribution(logits[verify_input.shape[1] - 1 - offset :])
        if append(sample(next_probs[0])):
            break
        proposal_probs = next_probs[1:]
        draft = sample(proposal_probs) if proposal_probs.shape[0] else None

    return ISDResult(prefix, proposed, accepted, forwards, processed_tokens)


class MapleCache(DynamicCache):
    """Full-history KV cache with matching packed-position and padding metadata.

    Sliding layers retain their history so speculative rollback remains valid;
    their attention masks still enforce Maple's window. Crop follows the pinned
    Transformers API: negative values remove that many trailing tokens.
    """

    def __init__(self):
        super().__init__()
        self.positions = self.segments = self.valid = None

    def append_metadata(self, positions, valid):
        if self.positions is None:
            if self.get_seq_length():
                raise RuntimeError("KV cache is missing its position metadata")
            self.positions, self.valid = positions.clone(), valid.clone()
        else:
            if self.positions.shape[-1] != self.get_seq_length():
                raise RuntimeError("Discard the cache after an interrupted forward")
            self.positions = torch.cat((self.positions, positions), dim=-1)
            self.valid = torch.cat((self.valid, valid), dim=-1)
        self.segments = (self.positions == 0).cumsum(-1)
        return self.segments[:, -positions.shape[-1] :]

    def crop(self, tokens_to_remove):
        super().crop(tokens_to_remove)
        length = self.get_seq_length()
        self._map_metadata(lambda tensor: tensor[:, :length])

    def _map_metadata(self, operation):
        for name in ("positions", "segments", "valid"):
            tensor = getattr(self, name)
            if tensor is not None:
                setattr(self, name, operation(tensor))

    def reset(self):
        # DynamicLayer.reset() zeroes values without reducing its length.
        # Reinitialize the empty dynamic cache together with its metadata.
        super().__init__()
        self.positions = self.segments = self.valid = None

    def reorder_cache(self, beam_idx):
        super().reorder_cache(beam_idx)
        self._map_metadata(lambda tensor: tensor.index_select(0, beam_idx.to(tensor.device)))

    def batch_repeat_interleave(self, repeats):
        super().batch_repeat_interleave(repeats)
        self._map_metadata(lambda tensor: tensor.repeat_interleave(repeats, dim=0))

    def batch_select_indices(self, indices):
        super().batch_select_indices(indices)
        self._map_metadata(lambda tensor: tensor[indices])


@torch.no_grad()
def initialize_mask_token(model, token_id, seed=42):
    """Initialize a new padded-vocabulary row, preserving every existing token.

    Mirrors the authors' new-token noisy-mean initialization. Works on ordinary
    weights or FSDP2 row shards, including CPU offload and HSDP replicas. Invoke
    only after initial pretrained loading, never after a training-state resume.
    """
    from torch.distributed.tensor import DTensor

    if not 0 < token_id < model.config.vocab_size:
        raise ValueError("New MASK must follow existing vocabulary rows and fit the padded vocabulary")
    norms = []
    for index, weight in enumerate((model.get_input_embeddings().weight, model.get_output_embeddings().weight)):
        group, offset = None, 0
        if isinstance(weight, DTensor):
            local = weight.to_local()
            shard_dims = [dim for dim, placement in enumerate(weight.placements) if placement.is_shard()]
            if len(shard_dims) > 1 or any(weight.placements[dim].dim != 0 for dim in shard_dims):
                raise ValueError("Mask initialization requires FSDP2 row-sharded embeddings")
            if shard_dims:
                dim = shard_dims[0]
                mesh = weight.device_mesh
                offset = mesh.get_local_rank(dim) * math.ceil(weight.shape[0] / mesh.size(dim))
                group = mesh.get_group(dim)
        else:
            local = weight
        rows = max(0, min(local.shape[0], token_id - offset))
        total = local[:rows].float().sum(0)
        if group is not None:
            total = total.to(get_device_type())
            dist.all_reduce(total, group=group)
        mean = total.cpu() / token_id
        generator = torch.Generator().manual_seed(seed + index)
        row = mean + torch.randn(mean.shape, generator=generator) / math.sqrt(weight.shape[1])
        if offset <= token_id < offset + local.shape[0]:
            local[token_id - offset].copy_(row.to(local))
        norms.append(float(row.norm()))
        if model.config.tie_word_embeddings:
            break
    return norms


def linear_loss(hidden, weight, targets, implementation):
    hidden, targets = hidden.reshape(-1, hidden.shape[-1]), targets.reshape(-1)
    if implementation == "liger_kernel":
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

        # Sum / clamped count remains finite on an all-ignored local sample.
        loss = LigerFusedLinearCrossEntropyLoss(reduction="sum")(weight, hidden, targets)
    elif implementation == "eager":
        loss = F.cross_entropy(F.linear(hidden, weight).float(), targets, ignore_index=-100, reduction="sum")
    else:
        raise ValueError(f"Maple supports eager or liger_kernel loss, got {implementation!r}")
    return loss / (targets != -100).sum().clamp_min(1)


def make_causal_block_mask(
    positions, segments, valid, window, *, key_positions=None, key_segments=None, key_valid=None
):
    from torch.nn.attention.flex_attention import create_block_mask

    key_positions = positions if key_positions is None else key_positions
    key_segments = segments if key_segments is None else key_segments
    key_valid = valid if key_valid is None else key_valid
    batch, length = positions.shape
    key_length = key_positions.shape[-1]

    def mask_mod(b, h, q, k):
        qs, ks = q.clamp(max=length - 1), k.clamp(max=key_length - 1)
        delta = positions[b, qs] - key_positions[b, ks]
        allowed = (delta >= 0) & (segments[b, qs] == key_segments[b, ks]) & valid[b, qs] & key_valid[b, ks]
        if window is not None:
            allowed = allowed & (delta <= window)
        return allowed & (q < length) & (k < key_length)

    return create_block_mask(
        mask_mod, B=batch, H=None, Q_LEN=length, KV_LEN=key_length, device=str(positions.device), _compile=True
    )


@contextmanager
def load_maple_for_inference(checkpoint, *, config=None, attention="flex_attention", config_kwargs=None):
    """Load ternary QAT weights under a single-process torchrun runtime.

    Evaluation uses FlexAttention's I-DLM mask; decoding uses causal SDPA.
    Always release the process group, including when loading or inference fails.
    """
    get_torch_device().set_device(0)
    dist.init_process_group(backend=get_dist_comm_backend())
    try:
        if dist.get_world_size() != 1:
            raise ValueError("Maple inference requires torchrun --nproc-per-node=1")
        init_parallel_state_from_config(AcceleratorConfig(), name="base")
        model = build_foundation_model(
            config_path=config or checkpoint,
            weights_path=checkpoint,
            torch_dtype="bfloat16",
            init_device=get_device_type(),
            config_kwargs=config_kwargs or {},
            ops_implementation=OpsImplementationConfig(
                attn_implementation=attention,
                moe_implementation="fused_triton",
                qat_implementation="ternary",
                cross_entropy_loss_implementation="liger_kernel",
                rms_norm_implementation="liger_kernel",
                rotary_pos_emb_implementation="eager",
                swiglu_mlp_implementation="eager",
                load_balancing_loss_implementation="eager",
            ),
        ).eval()
        yield model
    finally:
        dist.destroy_process_group()
        clear_parallel_state()
