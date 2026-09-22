"""Generate Maple v5 training from the pinned public Maple source.

patchgen veomni.models.transformers.maple.maple_gpu_patch_gen_config -o veomni/models/transformers/maple/generated --diff
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from patchgen import PatchConfig
from torch import nn
from transformers.generation.utils import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from veomni.models.transformers.maple.configuration_maple import MapleConfig


config = PatchConfig(
    source_module="veomni.models.transformers.maple.upstream.modeling_maple",
    target_file="patched_modeling_maple_gpu.py",
    description="Maple I-DLM with checkpoint-compatible ternary QAT and training-capable kernels",
)
config.drop_import_names("MapleConfig", "flash_attention_forward", "ROPE_INIT_FUNCTIONS", "dynamic_rope_update")
config.add_import("veomni.models.transformers.maple.configuration_maple", names=["MapleConfig"])
config.add_import(
    "veomni.models.transformers.maple.runtime",
    names=["prepare_idlm_inputs", "make_idlm_attention_mask", "shifted_targets", "balanced_idlm_loss"],
)
config.add_import("veomni.ops.qat.ternary", names=["ternary_fake_quant_weight", "ternary_linear"])
config.add_import("transformers.modeling_utils", names=["ALL_ATTENTION_FUNCTIONS"])
config.add_import("transformers.modeling_layers", names=["GradientCheckpointingLayer"])
config.add_post_import_block("""
from veomni.ops.dispatch import OpsConfigSlot, OpSlot
veomni_qat = OpsConfigSlot("qat_implementation")
veomni_moe = OpsConfigSlot("moe_implementation")
veomni_ce = OpsConfigSlot("cross_entropy_loss_implementation")
veomni_rms_norm = OpSlot("rms_norm", "standard")
""")


@config.replace_class("MapleOutputWithPast")
@dataclass
class MapleOutputWithPast(ModelOutput):
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    aux_metrics: dict | None = None
    past_key_values: object | None = None


@config.replace_class("MapleRotaryEmbedding")
class MapleRotaryEmbedding(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        if config.rope_scaling is not None and config.rope_scaling.get("rope_type", "default") != "default":
            raise NotImplementedError("Maple currently supports the checkpoint's unscaled partial RoPE")
        dim = int(config.head_dim * config.partial_rotary_factor)
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, inputs, position_ids):
        angles = position_ids.float()[..., None] * self.inv_freq.float()
        angles = torch.cat((angles, angles), -1)
        return angles.cos().to(inputs.dtype), angles.sin().to(inputs.dtype), angles


@config.override_method("MapleRMSNorm.forward")
def rms_norm_forward(self, hidden_states):
    if veomni_rms_norm.use_non_eager_impl:
        return veomni_rms_norm(hidden_states, self.weight, self.variance_epsilon)
    normed = hidden_states.float() * torch.rsqrt(
        hidden_states.float().square().mean(-1, keepdim=True) + self.variance_epsilon
    )
    return self.weight * normed.to(hidden_states.dtype)


# Patch: MapleMLP.__init__
# 1. Retain the checkpoint's ternary recipe for the eager expert path.
@config.override_method("MapleMLP.__init__")
def mlp_init(self, config, intermediate_size):
    super().__init__()
    self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
    self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
    self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
    self.group_size = config.ternary_group_size
    # --- Patch.1 ---
    self.ternary_scheme = config.ternary_scheme
    # --- Patch.1 ---


# Patch: MapleMLP.forward
# 1. Use the configured ternary recipe for all three expert projections.
@config.override_method("MapleMLP.forward")
def mlp_forward(self, inputs):
    # --- Patch.1 ---
    options = dict(group_size=self.group_size, scheme=self.ternary_scheme, enabled=veomni_qat.value == "ternary")
    # --- Patch.1 ---
    gate = ternary_linear(inputs, self.gate_proj.weight, **options).clamp(max=7.0)
    up = ternary_linear(inputs, self.up_proj.weight, **options).clamp(-7.0, 7.0)
    return ternary_linear(F.silu(gate) * up, self.down_proj.weight, **options)


@config.add_helper
class MapleExperts(nn.Module):
    """Packed experts with concatenated gate/up rows, not interleaved rows."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gate_up_proj = nn.Parameter(
            torch.empty(config.num_experts, 2 * config.moe_intermediate_size, config.hidden_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(config.num_experts, config.hidden_size, config.moe_intermediate_size)
        )

    def forward(self, hidden_states, indices, weights):
        gate_up, down = self.gate_up_proj, self.down_proj
        if veomni_qat.value == "ternary":
            options = dict(group_size=self.config.ternary_group_size, scheme=self.config.ternary_scheme)
            gate_up = ternary_fake_quant_weight(gate_up, **options)
            down = ternary_fake_quant_weight(down, **options)
        if veomni_moe.value in ("fused_triton", "fused_quack"):
            if veomni_moe.value == "fused_quack":
                from veomni.ops.kernels.moe.quack_gemm import quack_gemm_fused_moe_forward as moe_kernel
            else:
                from veomni.ops.kernels.moe.group_gemm import group_gemm_fused_moe_forward as moe_kernel
            return moe_kernel(
                num_experts=self.config.num_experts,
                routing_weights=weights.to(hidden_states.dtype),
                selected_experts=indices,
                hidden_states=hidden_states,
                fc1_1_weight=None,
                fc1_2_weight=None,
                fc1_1_2_weight=gate_up,
                fc2_weight=down,
                swiglu_limit=7.0,
            )
        if veomni_moe.value != "eager":
            raise ValueError(f"Unsupported Maple MoE implementation: {veomni_moe.value}")
        output = torch.zeros_like(hidden_states, dtype=torch.float32)
        for expert_id in range(self.config.num_experts):
            tokens, slots = torch.where(indices == expert_id)
            gate_weight, up_weight = gate_up[expert_id].chunk(2, dim=0)
            gate = F.linear(hidden_states[tokens], gate_weight).clamp(max=7.0)
            up = F.linear(hidden_states[tokens], up_weight).clamp(-7.0, 7.0)
            expert_output = F.linear(F.silu(gate) * up, down[expert_id])
            output = output.index_add(0, tokens, expert_output.float() * weights[tokens, slots, None])
        return output.to(hidden_states.dtype)


@config.override_method("MapleSparseMoeBlock._setup_experts")
def setup_experts(self):
    self.experts = MapleExperts(self.config)


@config.override_method("MapleSparseMoeBlock.forward")
def moe_forward(self, hidden_states):
    indices, weights, router_logits = self.gate(hidden_states)
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    return self.experts(flat, indices, weights).reshape_as(hidden_states), router_logits


@config.override_method("MapleSparseMoeBlock.moe_infer")
def moe_infer(self, hidden_states, topk_ids, topk_weight):
    return self.experts(hidden_states, topk_ids, topk_weight)


# Patch: MapleAttention.forward
# 1. Quantize attention projections with the checkpoint's ternary recipe.
@config.override_method("MapleAttention.forward")
def attention_forward(
    self, hidden_states, attention_mask=None, position_embeddings=None, past_key_value=None, use_cache=False, **kwargs
):
    batch, length, _ = hidden_states.shape
    # --- Patch.1 ---
    options = dict(
        group_size=self.config.ternary_group_size,
        scheme=self.config.ternary_scheme,
        enabled=veomni_qat.value == "ternary",
    )
    # --- Patch.1 ---
    query = (
        ternary_linear(hidden_states, self.q_proj.weight, **options)
        .view(batch, length, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key = (
        ternary_linear(hidden_states, self.k_proj.weight, **options)
        .view(batch, length, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    value = (
        ternary_linear(hidden_states, self.v_proj.weight, **options)
        .view(batch, length, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    query, key = self.q_norm(query), self.k_norm(key)
    if self.sliding_window is not None or not self.config.nope_on_global_attention:
        query, key = apply_rotary_pos_emb(query, key, *position_embeddings[:2])
    if use_cache:
        key, value = past_key_value.update(key, value, self.layer_idx)
    mask = attention_mask["sliding_attention" if self.sliding_window is not None else "full_attention"]
    implementation = self.config._attn_implementation
    if implementation in ("eager", "sdpa"):
        key = key.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=1)
        value = value.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=1)
        # cuDNN SDPA returns nonzero values for fully masked query rows. Give
        # those rows one finite dummy key, then remove their output/gradient.
        # This also avoids an all-masked softmax in the backend's backward.
        visible = mask.any(dim=-1, keepdim=True)
        safe_mask = mask.clone()
        safe_mask[..., 0] |= ~visible.squeeze(-1)
        output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=safe_mask, dropout_p=0.0, scale=self.scaling
        )
        output = output.masked_fill(~visible, 0).transpose(1, 2)
    else:
        # VeOmni's facade preserves the supplied BlockMask and supports backward.
        output, _ = ALL_ATTENTION_FUNCTIONS[implementation](
            self,
            query,
            key,
            value,
            mask,
            scaling=self.scaling,
            dropout=0.0,
        )
    output = output.reshape(batch, length, -1)
    return ternary_linear(output, self.o_proj.weight, self.o_proj.bias, **options), None, past_key_value


@config.replace_class("MapleDecoderLayer")
class MapleDecoderLayer(GradientCheckpointingLayer):
    """Keep checkpointing inside the layer so activation offload can wrap it."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = MapleAttention(config=config, layer_idx=layer_idx)
        self.mlp = MapleSparseMoeBlock(config)
        self.input_layernorm = MapleRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MapleRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        output_router_logits=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        attention, attention_weights, present = self.self_attn(
            hidden_states=self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + attention
        expert_output, router_logits = self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states + expert_output, attention_weights, present, 0.0, router_logits


@config.replace_class("MaplePreTrainedModel")
class MaplePreTrainedModel(PreTrainedModel):
    config_class = MapleConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MapleDecoderLayer"]
    _supports_attention_backend = True
    _supports_flex_attn = True
    _supports_sdpa = True

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, MapleGate)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, MapleRMSNorm):
            nn.init.ones_(module.weight)
        elif isinstance(module, MapleExperts):
            nn.init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            nn.init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)


@config.override_method("MapleModel.forward")
def model_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    inputs_embeds=None,
    use_cache=False,
    past_key_values=None,
    **kwargs,
):
    from veomni.models.transformers.maple.runtime import MapleCache

    if past_key_values is not None and not use_cache:
        raise ValueError("A supplied KV cache requires use_cache=True")
    if use_cache:
        if self.training or isinstance(attention_mask, dict):
            raise ValueError("Maple KV caching is only supported for causal evaluation")
        if past_key_values is None:
            past_key_values = MapleCache()
        elif not isinstance(past_key_values, MapleCache):
            raise ValueError("Use MapleCache to preserve position and padding metadata during rollback")
    if inputs_embeds is None:
        inputs_embeds = self.word_embeddings(input_ids)
    batch, length, _ = inputs_embeds.shape
    past_length = past_key_values.get_seq_length() if use_cache else 0
    if position_ids is None:
        position_ids = torch.arange(past_length, past_length + length, device=inputs_embeds.device)[None].expand(
            batch, -1
        )
    if not isinstance(attention_mask, dict):
        valid = torch.ones_like(position_ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
        if valid.shape[-1] != length:
            if not use_cache or valid.shape[-1] != past_length + length:
                raise ValueError("Attention mask must cover the query or complete cached sequence")
            valid = valid[:, -length:]
        if use_cache:
            segments = past_key_values.append_metadata(position_ids, valid)
            key_positions, key_segments, key_valid = (
                past_key_values.positions,
                past_key_values.segments,
                past_key_values.valid,
            )
        else:
            segments = (position_ids == 0).cumsum(-1)
            key_positions, key_segments, key_valid = position_ids, segments, valid
        # Non-I-DLM forwards are the causal verification/AR path.
        masks = {}
        for name, window in (("full_attention", None), ("sliding_attention", self.config.sliding_window)):
            if "flex" in self.config._attn_implementation:
                from veomni.models.transformers.maple.runtime import make_causal_block_mask

                masks[name] = make_causal_block_mask(
                    position_ids,
                    segments,
                    valid,
                    window,
                    key_positions=key_positions,
                    key_segments=key_segments,
                    key_valid=key_valid,
                )
            else:
                delta = position_ids[:, :, None] - key_positions[:, None, :]
                mask = (delta >= 0) & (segments[:, :, None] == key_segments[:, None, :])
                mask = mask & valid[:, :, None] & key_valid[:, None, :]
                if window is not None:
                    mask = mask & (delta <= window)
                masks[name] = mask[:, None]
        attention_mask = masks
    hidden = inputs_embeds
    positions = self.rotary_emb(hidden, position_ids)
    for layer in self.layers:
        outputs = layer(
            hidden,
            attention_mask=attention_mask,
            position_embeddings=positions,
            past_key_value=past_key_values,
            use_cache=use_cache,
        )
        hidden = outputs[0]
    return MoeModelOutputWithPast(last_hidden_state=self.norm(hidden), past_key_values=past_key_values)


# Patch: MapleForCausalLM
# 1. Normalize supervised loss sums by the trainer's unshifted response-token
#    count so packing and data-parallel partitioning preserve the objective.
@config.replace_class("MapleForCausalLM")
class MapleForCausalLM(MaplePreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.word_embeddings.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = MapleModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.word_embeddings

    def set_input_embeddings(self, embeddings):
        self.model.word_embeddings = embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, embeddings):
        self.lm_head = embeddings

    def forward(
        self,
        input_ids,
        labels=None,
        position_ids=None,
        attention_mask=None,
        use_cache=False,
        past_key_values=None,
        **kwargs,
    ):
        from veomni.models.transformers.maple.runtime import linear_loss

        if labels is not None and (use_cache or past_key_values is not None):
            raise ValueError("Supervised Maple forwards do not support KV caching")
        if position_ids is None:
            past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(past_length, past_length + input_ids.shape[-1], device=input_ids.device)[
                None
            ].expand_as(input_ids)
        metrics = None
        noisy_targets = None
        if labels is not None:
            targets = shifted_targets(labels, position_ids)
            if self.config.idlm_enabled:
                valid = (
                    torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
                )
                masks = {}
                for name, window in (("full_attention", None), ("sliding_attention", self.config.sliding_window)):
                    masks[name] = make_idlm_attention_mask(
                        position_ids,
                        valid,
                        self.config.idlm_block_size,
                        sliding_window=window,
                        flex="flex" in self.config._attn_implementation,
                    )
                input_ids, position_ids, noisy_targets, targets = prepare_idlm_inputs(
                    input_ids,
                    labels,
                    position_ids,
                    self.config.mask_token_id,
                )
                attention_mask = masks
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )
        hidden = outputs.last_hidden_state
        if labels is None:
            return MapleOutputWithPast(logits=self.lm_head(hidden), past_key_values=outputs.past_key_values)
        # --- Patch.1 ---
        normalizer = (labels != -100).sum().clamp_min(1)
        clean_count = (targets != -100).sum()
        if noisy_targets is None:
            loss = linear_loss(hidden, self.lm_head.weight, targets, veomni_ce.value) * clean_count / normalizer
        else:
            noisy_hidden, clean_hidden = hidden.chunk(2, dim=1)
            masked_loss = linear_loss(noisy_hidden, self.lm_head.weight, noisy_targets, veomni_ce.value)
            clean_loss = linear_loss(clean_hidden, self.lm_head.weight, targets, veomni_ce.value)
            loss = balanced_idlm_loss(
                masked_loss * (noisy_targets != -100).sum() / normalizer,
                clean_loss * clean_count / normalizer,
                self.config.idlm_clean_weight,
                self.config.idlm_auto_balance,
            )
            metrics = {"idlm_masked_ce": masked_loss.detach(), "idlm_clean_ce": clean_loss.detach()}
        # --- Patch.1 ---
        return MapleOutputWithPast(loss=loss, aux_metrics=metrics)
