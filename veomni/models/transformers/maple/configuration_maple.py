"""Maple architecture and I-DLM training configuration."""

from .upstream.configuration_maple import MapleConfig as ReferenceMapleConfig


class MapleConfig(ReferenceMapleConfig):
    model_type = "maple"

    def __init__(
        self,
        idlm_enabled=False,
        idlm_block_size=1,
        idlm_clean_weight=0.2,
        idlm_auto_balance=False,
        mask_token_id=None,
        idlm_initialize_mask_token=False,
        ternary_group_size=128,
        ternary_scheme="group_absmax",
        expert_weight_layout="packed_gate_up",
        sliding_window=512,
        layer_types=None,
        nope_on_global_attention=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.idlm_enabled = idlm_enabled
        self.idlm_block_size = idlm_block_size
        self.idlm_clean_weight = idlm_clean_weight
        self.idlm_auto_balance = idlm_auto_balance
        self.mask_token_id = mask_token_id
        self.idlm_initialize_mask_token = idlm_initialize_mask_token
        self.ternary_group_size = ternary_group_size
        self.ternary_scheme = ternary_scheme
        if expert_weight_layout != "packed_gate_up":
            raise ValueError("Maple requires expert_weight_layout=packed_gate_up; convert legacy weights on load")
        self.expert_weight_layout = expert_weight_layout
        self.sliding_window = sliding_window
        self.nope_on_global_attention = nope_on_global_attention
        self.layer_types = layer_types or [
            "full_attention" if (i + 1) % 4 == 0 else "sliding_attention" for i in range(self.num_hidden_layers)
        ]

    def validate_build_prerequisites(self):
        from veomni.distributed.parallel_state import get_parallel_state, is_parallel_state_initialized
        from veomni.ops.config.singleton import get_ops_config

        if self.idlm_block_size < 1 or self.idlm_clean_weight < 0:
            raise ValueError("Invalid I-DLM block size or clean loss weight")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("Maple layer_types must name every decoder layer")
        if self.idlm_enabled and (self.mask_token_id is None or not 0 <= self.mask_token_id < self.vocab_size):
            raise ValueError("I-DLM needs an in-vocabulary mask_token_id")
        if self.ternary_scheme not in ("group_absmax", "row_twn"):
            raise ValueError("Maple ternary_scheme must be group_absmax or row_twn")
        if self.ternary_scheme == "group_absmax":
            if self.ternary_group_size <= 0 or self.ternary_group_size & (self.ternary_group_size - 1):
                raise ValueError("Maple ternary_group_size must be a positive power of two")
            if self.hidden_size % self.ternary_group_size or self.moe_intermediate_size % self.ternary_group_size:
                raise ValueError(
                    "Maple hidden and expert intermediate dimensions must be divisible by ternary_group_size"
                )
        if is_parallel_state_initialized():
            state = get_parallel_state()
            if state.sp_enabled or state.any_extra_parallel_enabled:
                raise NotImplementedError("Maple currently supports FSDP2 data sharding; set SP/EP sizes to 1")
        ops = get_ops_config()
        if ops is not None:
            if ops.qat_implementation not in ("none", "ternary"):
                raise ValueError("Maple supports only none or ternary QAT")
            if ops.moe_implementation not in ("eager", "fused_triton", "fused_quack"):
                raise ValueError("Maple MoE supports eager, fused_triton, and fused_quack")
            if ops.attn_implementation not in ("eager", "sdpa", "flex_attention", "veomni_flex_attention_with_sp"):
                raise ValueError("Maple requires SDPA/eager reference attention or FlexAttention for I-DLM masks")
