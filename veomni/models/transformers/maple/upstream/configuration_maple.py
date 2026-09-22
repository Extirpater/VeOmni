"""Configuration for Maple models."""

from transformers.configuration_utils import PretrainedConfig


class MapleConfig(PretrainedConfig):
    """Configuration for the Maple mixture-of-experts causal language model."""

    model_type = "maple"

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=2048,
        num_hidden_layers=20,
        num_attention_heads=16,
        num_key_value_heads=4,
        hidden_act="silu",
        use_bias=False,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        attention_dropout=0.0,
        initializer_range=0.02,
        max_position_embeddings=32768,
        rope_theta=10000.0,
        use_cache=True,
        rope_scaling=None,
        partial_rotary_factor=0.5,
        pad_token_id=None,
        eos_token_id=None,
        num_experts=256,
        num_experts_per_tok=8,
        moe_intermediate_size=512,
        head_dim=128,
        output_router_logits=False,
        **kwargs,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.use_bias = use_bias
        self.rms_norm_eps = rms_norm_eps
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.use_cache = use_cache
        self.head_dim = head_dim or self.hidden_size // self.num_attention_heads
        self.rope_scaling = rope_scaling
        self.partial_rotary_factor = partial_rotary_factor

        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.output_router_logits = output_router_logits

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
