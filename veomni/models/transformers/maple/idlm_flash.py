"""FlashAttention-3 execution of Maple's I-DLM training masks.

Block size 1: the dual-stream mask from ``runtime.idlm_mask_mod`` factors into
dense varlen pieces over packed documents:

* clean query ``i`` sees clean keys ``j <= i`` (causal);
* noisy query ``i`` sees clean keys ``j < i`` (strictly causal) and its own
  noisy key, and nothing else.

Both streams keep Maple's window ``i - j <= sliding_window``. The strictly
causal term runs as a varlen call whose key length is one shorter than the
query length, so bottom-right causal alignment excludes the diagonal. The noisy
self key is a single dot product merged by log-sum-exp. Backward reuses FA3's
kernel with the merged output and LSE, which yields exactly the gradients of
the combined softmax restricted to the clean keys; the self term is closed form.

Block sizes B > 1 (paper stride N = B + 1): noisy query at document position
p = j*B + r sees clean keys at positions < j*B and noisy keys j*B + t for
t <= r. Ordering each document's noisy tokens by residue r = p mod B (done once
at the model input; every other layer is per-token) makes the clean part a set
of B*B strided varlen calls: for query residue r and key residue s, strided
query j sees strided key m iff m*B + s < j*B, i.e. m <= j - 1 for every s. The
window i - k <= W becomes m >= j - floor((W + s - r) / B). The at most B
in-block noisy keys are merged in closed form, as the self key is for B = 1.

Documents come from ``position_ids == 0``. The collator gives every pad token
position 0 and a true attention mask, so pads are isolated singleton documents,
matching the tile-level flex mask. Padding marked invalid is not supported.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from ....utils.seqlen_pos_transform_utils import pos2culen


def uses_flash_attention_3(config):
    return "flash_attention_3" in (config._attn_implementation or "")


def _documents(position_ids, valid_tokens):
    """Return int32 ``cu_seqlens``, document lengths, and the maximum length of one packed row."""
    if position_ids.shape[0] != 1:
        raise ValueError("Maple FA3 attention expects one packed row per microbatch")
    positions = position_ids[0]
    cu_seqlens = pos2culen(positions)
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).long()
    # FA3 needs the maximum length on the host; fetch the input checks in the same sync.
    all_valid = valid_tokens.all() if valid_tokens is not None else positions.new_ones((), dtype=torch.bool)
    longest = torch.cat((lengths, lengths.new_zeros(1))).max()
    max_seqlen, starts_document, valid = torch.stack((longest, positions[0] == 0, all_valid)).tolist()
    if not starts_document:
        raise ValueError("Maple FA3 attention needs each packed row to start a document at position 0")
    if not valid:
        raise ValueError("Maple FA3 attention requires isolated pads, not invalid attention-mask tokens")
    return cu_seqlens, lengths, int(max_seqlen)


# The raw ops write into caller buffers; the Python custom-op wrappers forbid that aliasing.
def _fa3_forward(q, k, v, out, cu_q, cu_k, seqused_k, max_q, max_k, left, scale):
    import flash_attn_interface

    result, lse, *_ = flash_attn_interface.flash_attn_3_gpu.fwd(
        q, k, v, None, None, None, out, cu_q, cu_k, None, None, seqused_k,
        max_q, max_k, None, None, None, None, None, None, None, None, None,
        scale, True, left, 0, 0, 0.0, True, None, 1, None, 0,
    )  # fmt: skip
    return result, lse


def _fa3_backward(dout, q, k, v, out, lse, dq, dk, dv, cu_q, cu_k, seqused_k, max_q, max_k, left, scale):
    import flash_attn_interface

    flash_attn_interface.flash_attn_3_gpu.bwd(
        dout, q, k, v, out, lse, dq, dk, dv, cu_q, cu_k, None, seqused_k,
        max_q, max_k, scale, True, left, 0, 0.0, False, 0,
    )  # fmt: skip


def _repeat_heads(states, groups):
    return states.repeat_interleave(groups, dim=1).float()


def _merge(parts, q, keys, values, scale: float):
    """Merge FA3 partial results with directly attended keys by log-sum-exp.

    ``parts`` holds ``(out [T, H, D], lse [H, T])``; ``keys``/``values`` are
    ``[T, Hkv, D]``, one key per query. Returns BF16 output and FP32 LSE ``[H, T]``.
    """
    groups = q.shape[1] // keys[0].shape[1]
    lses = [lse.transpose(0, 1) for _, lse in parts]
    scores = [(q.float() * _repeat_heads(key, groups)).sum(-1) * scale for key in keys]
    total = torch.logsumexp(torch.stack(lses + scores), dim=0)
    out = sum(torch.exp(lse - total)[..., None] * part.float() for (part, _), lse in zip(parts, lses))
    out = out + sum(torch.exp(s - total)[..., None] * _repeat_heads(v, groups) for s, v in zip(scores, values))
    return out.to(q.dtype), total.transpose(0, 1).contiguous()


def _direct_key_grads(dout, out, lse, q, keys, values, scale: float):
    """Closed-form FP32 gradients for directly attended keys under the merged softmax."""
    tokens, heads, dim = q.shape
    kv_heads = keys[0].shape[1]
    groups = heads // kv_heads
    dout, qf = dout.float(), q.float()
    delta = (dout * out.float()).sum(-1)
    lse = lse.transpose(0, 1)
    dq = torch.zeros_like(qf)
    grads = []
    for key, value in zip(keys, values):
        kr, vr = _repeat_heads(key, groups), _repeat_heads(value, groups)
        prob = torch.exp((qf * kr).sum(-1) * scale - lse)
        dscore = prob * ((dout * vr).sum(-1) - delta) * scale
        dq = dq + dscore[..., None] * kr
        dk = (dscore[..., None] * qf).view(tokens, kv_heads, groups, dim).sum(2)
        dv = (prob[..., None] * dout).view(tokens, kv_heads, groups, dim).sum(2)
        grads.append((dk, dv))
    return dq, grads


# Block size 1 keeps the padded row length; residue-group sizes vary with documents.
_merge_static = torch.compile(_merge, fullgraph=True, dynamic=False)
_merge_dynamic = torch.compile(_merge, fullgraph=True, dynamic=True)
_direct_key_grads_static = torch.compile(_direct_key_grads, fullgraph=True, dynamic=False)
_direct_key_grads_dynamic = torch.compile(_direct_key_grads, fullgraph=True, dynamic=True)


# ----------------------------------------------------------------------------
# Block size 1
# ----------------------------------------------------------------------------


@dataclass
class IDLMFlashMask:
    """Varlen metadata for one packed row of length ``length`` per stream."""

    cu_seqlens: torch.Tensor
    strict_seqused: torch.Tensor
    max_seqlen: int
    length: int
    window: Optional[int]
    idlm: bool = True

    def left(self, strict):
        if self.window is None:
            return -1
        return self.window - 1 if strict else self.window

    def forward(self, q, k, v, scale, out=None, strict=False):
        seqused = self.strict_seqused if strict else None
        cu, most = self.cu_seqlens, self.max_seqlen
        return _fa3_forward(q, k, v, out, cu, cu, seqused, most, most, self.left(strict), scale)

    def backward(self, dout, q, k, v, out, lse, dq, dk, dv, scale, strict=False):
        seqused = self.strict_seqused if strict else None
        cu, most = self.cu_seqlens, self.max_seqlen
        _fa3_backward(dout, q, k, v, out, lse, dq, dk, dv, cu, cu, seqused, most, most, self.left(strict), scale)


def make_idlm_flash_masks(position_ids, valid_tokens, sliding_window, *, idlm=True):
    """Build full- and sliding-attention metadata shared by every layer."""
    cu_seqlens, lengths, max_seqlen = _documents(position_ids, valid_tokens)
    strict = (lengths - 1).to(torch.int32)
    length = position_ids.shape[-1]
    return {
        name: IDLMFlashMask(cu_seqlens, strict, max_seqlen, length, window, idlm)
        for name, window in (("full_attention", None), ("sliding_attention", sliding_window))
    }


class _IDLMFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, mask, scale):
        out = torch.empty_like(q)
        if mask.idlm:
            length = mask.length
            qn, kn, vn = q[:length], k[:length], v[:length]
            _, lse_clean = mask.forward(q[length:], k[length:], v[length:], scale, out=out[length:])
            strict = mask.forward(qn, k[length:], v[length:], scale, strict=True)
            noisy_out, lse_noisy = _merge_static([strict], qn, [kn], [vn], scale)
            out[:length].copy_(noisy_out)
            ctx.save_for_backward(q, k, v, out, lse_noisy, lse_clean)
        else:
            _, lse_clean = mask.forward(q, k, v, scale, out=out)
            ctx.save_for_backward(q, k, v, out, lse_clean)
        ctx.mask, ctx.scale = mask, scale
        return out

    @staticmethod
    def backward(ctx, dout):
        mask, scale = ctx.mask, ctx.scale
        dout = dout.contiguous()
        if not mask.idlm:
            q, k, v, out, lse = ctx.saved_tensors
            dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
            mask.backward(dout, q, k, v, out, lse, dq, dk, dv, scale)
            return dq, dk, dv, None, None
        q, k, v, out, lse_noisy, lse_clean = ctx.saved_tensors
        n = mask.length
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        mask.backward(dout[n:], q[n:], k[n:], v[n:], out[n:], lse_clean, dq[n:], dk[n:], dv[n:], scale)
        # Keys past each document's strict length receive no FA3 write.
        dk_strict, dv_strict = torch.zeros_like(k[n:]), torch.zeros_like(v[n:])
        mask.backward(
            dout[:n], q[:n], k[n:], v[n:], out[:n], lse_noisy, dq[:n], dk_strict, dv_strict, scale, strict=True
        )
        dk[n:] += dk_strict
        dv[n:] += dv_strict
        dq_self, [(dk_self, dv_self)] = _direct_key_grads_static(
            dout[:n], out[:n], lse_noisy, q[:n], [k[:n]], [v[:n]], scale
        )
        dq[:n] += dq_self.to(dq.dtype)
        dk[:n], dv[:n] = dk_self.to(dk.dtype), dv_self.to(dv.dtype)
        return dq, dk, dv, None, None


def idlm_flash_attention(query, key, value, mask: IDLMFlashMask, scale: float):
    """Attend ``[tokens, heads, head_dim]`` tensors of one packed row."""
    return _IDLMFlashAttention.apply(query, key, value, mask, scale)


# ----------------------------------------------------------------------------
# Block sizes B > 1
# ----------------------------------------------------------------------------


@dataclass
class IDLMFlashBlockMask:
    """Metadata for one packed row with residue-ordered noisy tokens."""

    block_size: int
    length: int
    window: Optional[int]
    # Causal metadata for the clean stream (original order).
    clean: IDLMFlashMask
    # noisy_order[i] = original noisy index placed at residue-ordered index i.
    noisy_order: torch.Tensor
    group_offsets: tuple  # start of residue group r in residue order, length B + 1
    group_cu: tuple  # per residue, int32 [docs + 1] cumulative counts within the group
    group_strict: tuple  # per residue, int32 [docs] strict key counts (count - 1, >= 0)
    group_max: int
    # self_index[r][t] (t < r): residue-order index of the in-block noisy key t per query of group r.
    self_index: tuple

    def rows(self, r):
        return slice(self.group_offsets[r], self.group_offsets[r + 1])

    def left(self, r, s):
        """Strided left window, or ``None`` when no key of residue ``s`` is in range.

        FA3 reads -1 as an unlimited window, so empty ranges must be skipped.
        """
        if self.window is None:
            return -1
        left = (self.window + s - r) // self.block_size - 1
        return left if left >= 0 else None

    def group_args(self, r, s):
        return self.group_cu[r], self.group_cu[s], self.group_strict[r], self.group_max, self.group_max


def make_idlm_flash_block_masks(position_ids, valid_tokens, sliding_window, block_size):
    if sliding_window is not None and sliding_window < block_size:
        raise ValueError("Maple FA3 block attention requires sliding_window >= idlm_block_size")
    cu_seqlens, lengths, max_seqlen = _documents(position_ids, valid_tokens)
    positions = position_ids[0]
    length = positions.numel()
    device = positions.device
    noisy_order = torch.sort(positions % block_size, stable=True).indices  # (residue, document, position)
    counts = torch.stack([((lengths - r + block_size - 1) // block_size).clamp_min(0) for r in range(block_size)])
    offsets = [0]
    for total in counts.sum(-1).tolist():
        offsets.append(offsets[-1] + total)
    group_cu = torch.zeros(block_size, counts.shape[1] + 1, dtype=torch.int32, device=device)
    group_cu[:, 1:] = counts.cumsum(-1)
    starts = group_cu[:, :-1].long()
    documents = torch.arange(counts.shape[1], device=device)
    self_index = []
    for r in range(block_size):
        doc = torch.repeat_interleave(documents, counts[r])
        within = torch.arange(offsets[r + 1] - offsets[r], device=device) - starts[r, doc]
        self_index.append(tuple(offsets[t] + starts[t, doc] + within for t in range(r)))
    shared = dict(
        block_size=block_size,
        length=length,
        noisy_order=noisy_order,
        group_offsets=tuple(offsets),
        group_cu=tuple(group_cu.unbind(0)),
        group_strict=tuple((counts - 1).clamp_min(0).to(torch.int32).unbind(0)),
        group_max=(max_seqlen + block_size - 1) // block_size,
        self_index=tuple(self_index),
    )
    return {
        name: IDLMFlashBlockMask(
            window=window, clean=IDLMFlashMask(cu_seqlens, None, max_seqlen, length, window, False), **shared
        )
        for name, window in (("full_attention", None), ("sliding_attention", sliding_window))
    }


def _direct_keys(mask, r, kn, vn):
    """In-block noisy keys/values for residue group ``r``: earlier residues, then the query itself."""
    rows = mask.rows(r)
    keys = [kn[index] for index in mask.self_index[r]] + [kn[rows]]
    values = [vn[index] for index in mask.self_index[r]] + [vn[rows]]
    return keys, values


class _IDLMFlashBlockAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, mask, scale):
        n, block = mask.length, mask.block_size
        out = torch.empty_like(q)
        _, lse_clean = mask.clean.forward(q[n:], k[n:], v[n:], scale, out=out[n:])
        # Clean keys in residue order, shared by every noisy group and reused in backward.
        kc, vc = k[n:][mask.noisy_order], v[n:][mask.noisy_order]
        qn, kn, vn = q[:n], k[:n], v[:n]
        lses = []
        for r in range(block):
            rows = mask.rows(r)
            parts = []
            for s in range(block):
                left = mask.left(r, s)
                if left is not None:
                    keys = mask.rows(s)
                    parts.append(_fa3_forward(qn[rows], kc[keys], vc[keys], None, *mask.group_args(r, s), left, scale))
            merged, lse = _merge_dynamic(parts, qn[rows], *_direct_keys(mask, r, kn, vn), scale)
            out[rows].copy_(merged)
            lses.append(lse)
        ctx.save_for_backward(q, k, v, out, kc, vc, lse_clean, *lses)
        ctx.mask, ctx.scale = mask, scale
        return out

    @staticmethod
    def backward(ctx, dout):
        mask, scale = ctx.mask, ctx.scale
        q, k, v, out, kc, vc, lse_clean, *lses = ctx.saved_tensors
        n, block = mask.length, mask.block_size
        dout = dout.contiguous()
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        mask.clean.backward(dout[n:], q[n:], k[n:], v[n:], out[n:], lse_clean, dq[n:], dk[n:], dv[n:], scale)
        qn, kn, vn = q[:n], k[:n], v[:n]
        dkc, dvc = torch.zeros_like(kc, dtype=torch.float32), torch.zeros_like(vc, dtype=torch.float32)
        dqn = torch.zeros_like(qn, dtype=torch.float32)
        dkn = torch.zeros_like(kn, dtype=torch.float32)
        dvn = torch.zeros_like(vn, dtype=torch.float32)
        for r in range(block):
            rows = mask.rows(r)
            q_r, o_r, do_r, lse_r = qn[rows], out[:n][rows], dout[:n][rows], lses[r]
            for s in range(block):
                left = mask.left(r, s)
                if left is None:
                    continue
                keys = mask.rows(s)
                dq_part = torch.empty_like(q_r)
                # Keys past each document's strict length receive no FA3 write.
                dk_part, dv_part = torch.zeros_like(kc[keys]), torch.zeros_like(vc[keys])
                _fa3_backward(
                    do_r, q_r, kc[keys], vc[keys], o_r, lse_r, dq_part, dk_part, dv_part,
                    *mask.group_args(r, s), left, scale,
                )  # fmt: skip
                dqn[rows] += dq_part
                dkc[keys] += dk_part
                dvc[keys] += dv_part
            keys, values = _direct_keys(mask, r, kn, vn)
            dq_direct, grads = _direct_key_grads_dynamic(do_r, o_r, lse_r, q_r, keys, values, scale)
            dqn[rows] += dq_direct
            for index, (dk_t, dv_t) in zip(mask.self_index[r], grads[:-1]):
                dkn.index_add_(0, index, dk_t)
                dvn.index_add_(0, index, dv_t)
            dkn[rows] += grads[-1][0]
            dvn[rows] += grads[-1][1]
        dq[:n], dk[:n], dv[:n] = dqn.to(dq.dtype), dkn.to(dk.dtype), dvn.to(dv.dtype)
        # Return clean-key gradients from residue order to the original order.
        dk[n:] = dk[n:].float().index_add_(0, mask.noisy_order, dkc).to(dk.dtype)
        dv[n:] = dv[n:].float().index_add_(0, mask.noisy_order, dvc).to(dv.dtype)
        return dq, dk, dv, None, None


def idlm_flash_block_attention(query, key, value, mask: IDLMFlashBlockMask, scale: float):
    """Attend ``[tokens, heads, head_dim]`` with residue-ordered noisy tokens."""
    return _IDLMFlashBlockAttention.apply(query, key, value, mask, scale)
