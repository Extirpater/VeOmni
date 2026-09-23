"""Maple FA3 I-DLM attention, fused loss, router and QAT fast paths against references."""

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from veomni.models.transformers.maple.runtime import (
    linear_loss,
    make_idlm_attention_mask,
    maple_router_logits,
    weighted_linear_loss,
)
from veomni.ops.qat.ternary import ternary_fake_quant_weight
from veomni.ops.qat.ternary_twn import quantize_twn_native


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _relative(actual, expected):
    return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()


@pytest.mark.parametrize("window", [None, 5, 64])
@pytest.mark.parametrize("idlm", [True, False])
def test_fa3_idlm_attention_matches_dense_mask(window, idlm):
    pytest.importorskip("flash_attn_interface")
    from veomni.models.transformers.maple.idlm_flash import idlm_flash_attention, make_idlm_flash_masks

    torch.manual_seed(0)
    heads, kv_heads, dim = 16, 4, 128
    # Singleton documents mirror collator pads (position 0, attention mask 1).
    positions = torch.cat([torch.arange(n) for n in (1, 300, 7, 1, 1, 450, 2)])[None].cuda()
    length = positions.shape[1]
    tokens = 2 * length if idlm else length
    valid = torch.ones_like(positions, dtype=torch.bool)
    if idlm:
        dense = make_idlm_attention_mask(positions, valid, 1, sliding_window=window)[0, 0]
    else:
        p, segments = positions[0], (positions[0] == 0).cumsum(0)
        delta = p[:, None] - p[None]
        dense = (delta >= 0) & (segments[:, None] == segments[None])
        if window is not None:
            dense &= delta <= window
    q = torch.randn(tokens, heads, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(tokens, kv_heads, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(tokens, kv_heads, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    scale = dim**-0.5
    mask = make_idlm_flash_masks(positions, valid, window, idlm=idlm)[
        "sliding_attention" if window else "full_attention"
    ]
    out = idlm_flash_attention(q, k, v, mask, scale)
    grad = torch.randn_like(out)
    actual = torch.autograd.grad(out, (q, k, v), grad)

    qf, kf, vf = (x.detach().float().requires_grad_() for x in (q, k, v))
    scores = torch.einsum("qhd,khd->hqk", qf, kf.repeat_interleave(heads // kv_heads, 1)) * scale
    probs = scores.masked_fill(~dense[None], float("-inf")).softmax(-1)
    ref = torch.einsum("hqk,khd->qhd", probs, vf.repeat_interleave(heads // kv_heads, 1))
    expected = torch.autograd.grad(ref, (qf, kf, vf), grad.float())
    assert _relative(out, ref) < 1e-2
    for a, e in zip(actual, expected):
        assert _relative(a, e) < 1e-2


def test_weighted_linear_loss_matches_two_stream_liger():
    pytest.importorskip("liger_kernel")
    torch.manual_seed(0)
    tokens, hidden, vocab, half = 4096, 256, 5000, 2048
    h = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = (torch.randn(vocab, hidden, device="cuda") * 0.05).bfloat16().requires_grad_()
    t = torch.randint(0, vocab, (tokens,), device="cuda")
    t[torch.rand(tokens, device="cuda") < 0.3] = -100
    noisy, clean = t[:half], t[half:]
    ref = linear_loss(h[:half], w, noisy, "liger_kernel") * (noisy != -100).sum()
    ref = ref + 0.2 * linear_loss(h[half:], w, clean, "liger_kernel") * (clean != -100).sum()
    expected = torch.autograd.grad(ref / 100, (h, w))
    total, per_token = weighted_linear_loss(h, w, t, ((half, 1.0), (tokens - half, 0.2)))
    actual = torch.autograd.grad(total / 100, (h, w))
    torch.testing.assert_close(total, ref.float(), rtol=1e-4, atol=1e-3)
    assert per_token[t == -100].abs().max() == 0
    for a, e in zip(actual, expected):
        assert _relative(a, e) < 1e-2


def test_router_logits_match_fp32_linear():
    torch.manual_seed(0)
    x = torch.randn(512, 2048, device="cuda").bfloat16()
    w = (torch.randn(256, 2048, device="cuda") * 0.02).bfloat16()
    out = maple_router_logits(x, w)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, torch.nn.functional.linear(x.float(), w.float()), rtol=1e-4, atol=1e-4)


def test_twn_recompute_reuses_forward_statistics_bitwise():
    torch.manual_seed(0)
    weight = (torch.randn(16, 1024, 2048, device="cuda") * 0.02).bfloat16().requires_grad_()
    cache, seen = {}, []

    def body(x):
        quantized = ternary_fake_quant_weight(weight, scheme="row_twn", stats_cache=cache)
        seen.append(quantized.detach().clone())
        return (x @ quantized[0].t()).sum()

    x = torch.randn(4, 2048, device="cuda").bfloat16().requires_grad_()
    checkpoint(body, x, use_reentrant=False).backward()
    expected = quantize_twn_native(weight.detach())
    assert len(seen) == 2 and all(torch.equal(q, expected) for q in seen)
    assert weight.grad is not None


@pytest.mark.parametrize("block_size", [1, 2, 3])
def test_fa3_model_matches_flex_loss_and_gradients(block_size):
    pytest.importorskip("flash_attn_interface")
    from veomni.models.auto import build_foundation_model
    from veomni.models.transformers.maple.configuration_maple import MapleConfig

    from ..tools.training_utils import make_eager_ops_config

    def build(attention):
        config = MapleConfig(
            vocab_size=256,
            hidden_size=128,
            head_dim=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=128,
            ternary_scheme="row_twn",
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=4,
            idlm_enabled=True,
            idlm_block_size=block_size,
            mask_token_id=255,
            architectures=["MapleForCausalLM"],
        )
        ops = make_eager_ops_config(
            qat_implementation="ternary",
            attn_implementation=attention,
            cross_entropy_loss_implementation="liger_kernel",
            rms_norm_implementation="liger_kernel",
        )
        model = build_foundation_model(config, torch_dtype="bfloat16", init_device="cpu", ops_implementation=ops)
        if next(model.parameters()).is_meta:
            model.to_empty(device="cpu")
        return model

    torch.manual_seed(0)
    lengths = [1, 37, 5, 1, 90, 2]
    ids = torch.randint(0, 250, (1, sum(lengths)))
    positions = torch.cat([torch.arange(n) for n in lengths])[None]
    labels = ids.clone()
    labels[:, :3] = -100
    labels[:, 50:60] = -100
    results = []
    state = None
    for attention in ("flex_attention", "flash_attention_3"):
        model = build(attention)
        if state is None:
            model.init_weights()
            state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            model.load_state_dict(state)
        model = model.cuda().train()
        loss = model(
            input_ids=ids.cuda(),
            labels=labels.cuda(),
            position_ids=positions.cuda(),
            attention_mask=torch.ones_like(ids).cuda(),
        ).loss
        loss.backward()
        results.append((loss.detach().float(), {n: p.grad.float() for n, p in model.named_parameters()}))
    (flex_loss, flex_grads), (fa3_loss, fa3_grads) = results
    torch.testing.assert_close(fa3_loss, flex_loss, rtol=2e-2, atol=2e-2)
    # BF16 attention differences can flip near-tied top-2 choices of the tiny
    # random router, so parameters on the routed path get a looser bound.
    routed = ("mlp.", "post_attention_layernorm")
    for name, expected in flex_grads.items():
        bound = 0.3 if any(part in name for part in routed) else 0.05
        assert _relative(fa3_grads[name], expected) < bound, name
