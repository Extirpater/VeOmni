import copy
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch import nn

from veomni.models.auto import build_foundation_model
from veomni.models.transformers.maple.configuration_maple import MapleConfig
from veomni.models.transformers.maple.runtime import (
    balanced_idlm_loss,
    introspective_generate,
    make_idlm_attention_mask,
    prepare_idlm_inputs,
)
from veomni.ops.qat.ternary import ternary_fake_quant_weight, ternary_linear

from ..tools.training_utils import make_eager_ops_config


def make_model(*, gpu=False, fused=False):
    config = MapleConfig(
        vocab_size=256 if gpu else 32,
        hidden_size=128 if gpu else 16,
        head_dim=64 if gpu else 8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=128 if gpu else 16,
        ternary_group_size=8,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=2,
        idlm_enabled=True,
        mask_token_id=31,
        architectures=["MapleForCausalLM"],
    )
    overrides = dict(qat_implementation="ternary")
    if fused:
        overrides.update(
            attn_implementation="flex_attention",
            moe_implementation="fused_triton",
            cross_entropy_loss_implementation="liger_kernel",
            rms_norm_implementation="liger_kernel",
        )
    ops = make_eager_ops_config(**overrides)
    model = build_foundation_model(
        config, torch_dtype="bfloat16" if gpu else "float32", init_device="cpu", ops_implementation=ops
    )
    # The production loader meta-initializes; a small standalone test materializes.
    if next(model.parameters()).is_meta:
        model.to_empty(device="cpu")
    model.init_weights()
    if gpu:
        model = model.to(device="cuda")
    return model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Maple inference runtime uses CUDA kernels")
@pytest.mark.parametrize("attention", ["flex_attention", "sdpa"])
@pytest.mark.parametrize("interrupt", [False, True])
def test_inference_runtime_roundtrip_and_cleanup(tmp_path, monkeypatch, free_tcp_port, attention, interrupt):
    import torch.distributed as dist

    from veomni.distributed.parallel_state import is_parallel_state_initialized
    from veomni.models.transformers.maple.runtime import load_maple_for_inference

    for name, value in {
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(free_tcp_port),
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
    }.items():
        monkeypatch.setenv(name, value)
    model = make_model(gpu=True, fused=True).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]], device="cuda")
    with torch.no_grad():
        expected = model(input_ids=ids).logits
    model.save_pretrained(tmp_path)
    del model

    class InferenceInterrupted(Exception):
        pass

    try:
        with load_maple_for_inference(str(tmp_path), attention=attention) as loaded, torch.no_grad():
            actual = loaded(input_ids=ids).logits
            torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
            if interrupt:
                raise InferenceInterrupted
    except InferenceInterrupted:
        pass
    assert not dist.is_initialized()
    assert not is_parallel_state_initialized()


def test_clean_branch_equals_causal_ar_and_has_no_noisy_leakage():
    model = make_model().eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    pos = torch.arange(6)[None]
    inputs, doubled, _, _ = prepare_idlm_inputs(ids, ids, pos, 31)
    masks = {
        name: make_idlm_attention_mask(pos, torch.ones_like(ids), 2, sliding_window=window)
        for name, window in [("full_attention", None), ("sliding_attention", 2)]
    }
    clean = model.model(input_ids=ids).last_hidden_state
    both = model.model(input_ids=inputs, position_ids=doubled, attention_mask=masks).last_hidden_state
    torch.testing.assert_close(both[:, 6:], clean, atol=1e-6, rtol=1e-5)
    inputs[:, :6] = 7
    changed = model.model(input_ids=inputs, position_ids=doubled, attention_mask=masks).last_hidden_state
    torch.testing.assert_close(changed[:, 6:], clean, atol=1e-6, rtol=1e-5)


def test_training_backward_checkpointing_and_roundtrip(tmp_path):
    torch.manual_seed(20)
    model = make_model().train()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    labels = ids.clone()
    labels[:, :2] = -100
    baseline = copy.deepcopy(model)
    expected = baseline(input_ids=ids, labels=labels).loss
    expected.backward()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    output = model(input_ids=ids, labels=labels)
    assert set(output.aux_metrics) == {"idlm_masked_ce", "idlm_clean_ce"}
    torch.testing.assert_close(output.loss, expected)
    output.loss.backward()
    for (name, param), (_, ref) in zip(model.named_parameters(), baseline.named_parameters()):
        assert param.grad is not None, name
        torch.testing.assert_close(param.grad, ref.grad, rtol=1e-5, atol=1e-6)
    assert model.model.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.step()
    model.save_pretrained(tmp_path)
    reloaded = make_model()
    from safetensors.torch import load_file

    reloaded.load_state_dict(load_file(tmp_path / "model.safetensors"), strict=True)
    reloaded.eval()
    model.eval()
    torch.testing.assert_close(reloaded(input_ids=ids).logits, model(input_ids=ids).logits)
    assert "model.layers.0.mlp.experts.0.gate_proj.weight" in reloaded.state_dict()
    result = introspective_generate(reloaded, ids, mask_token_id=31, max_new_tokens=4)
    assert result.sequences.shape == (1, 10)


def test_new_mask_initialization_preserves_existing_tokens():
    from veomni.models.transformers.maple.runtime import initialize_mask_token

    model = make_model()
    previous = {name: value.clone() for name, value in model.state_dict().items()}
    twin = copy.deepcopy(model)
    initialize_mask_token(model, 31, seed=17)
    initialize_mask_token(twin, 31, seed=17)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, twin.state_dict()[name], atol=0, rtol=0)
        if name in ("model.word_embeddings.weight", "lm_head.weight"):
            torch.testing.assert_close(value[:31], previous[name][:31], atol=0, rtol=0)
            assert not torch.equal(value[31], previous[name][31])
        else:
            torch.testing.assert_close(value, previous[name], atol=0, rtol=0)


def test_dcp_export_keeps_registered_config_tokenizer_and_trained_mask(tmp_path, monkeypatch):
    import torch.distributed.checkpoint as dcp
    from safetensors.torch import load_file
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    from veomni.models.loader import get_model_config
    from veomni.utils.device import get_device_type

    if get_device_type() == "cpu":
        # The exporter loads tensors on CPU; accelerator cache cleanup is irrelevant here.
        monkeypatch.setattr("veomni.checkpoint.dcp_checkpointer.empty_cache", lambda: None)

    model = make_model().eval()
    # Deliberately distinctive trained rows must survive export without new-token initialization.
    with torch.no_grad():
        model.model.word_embeddings.weight[31].fill_(0.75)
        model.lm_head.weight[31].fill_(-0.5)
    assets, checkpoint, exported = (tmp_path / name for name in ("assets", "dcp", "export"))
    model.config.save_pretrained(assets)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({**{f"t{i}": i for i in range(31)}, "<mask>": 31}, unk_token="t0")),
        unk_token="t0",
        mask_token="<mask>",
    )
    tokenizer.chat_template = "{{ messages[0]['content'] }}"
    tokenizer.save_pretrained(assets)
    state = {name: tensor.to(torch.bfloat16) for name, tensor in model.state_dict().items()}
    dcp.save({"model": state}, checkpoint_id=checkpoint)
    merge = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/merge_dcp_to_hf.py"))
    merge["merge_to_hf_pt"](str(checkpoint), str(exported), str(assets))
    config = get_model_config(str(exported))
    assert isinstance(config, MapleConfig)
    assert config.idlm_enabled and config.ternary_group_size == 8 and config.mask_token_id == 31
    restored_tokenizer = AutoTokenizer.from_pretrained(exported)
    assert restored_tokenizer.mask_token_id == 31
    assert restored_tokenizer.chat_template == tokenizer.chat_template
    restored = load_file(exported / "model.safetensors")
    assert restored.keys() == state.keys()
    for name, tensor in state.items():
        torch.testing.assert_close(restored[name], tensor, atol=0, rtol=0)


@pytest.mark.parametrize("packed", [False, True])
@torch.no_grad()
def test_cached_chunks_and_rollback_match_full_causal_logits(packed):
    model = make_model().eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])
    positions = torch.arange(12)[None]
    if packed:
        positions[:, 6:] -= 6
    valid = torch.ones_like(ids)
    valid[:, 4] = 0
    expected = model(input_ids=ids, position_ids=positions, attention_mask=valid).logits
    cache = None
    outputs = []
    start = 0
    for stop in (4, 7, 12):
        output = model(
            input_ids=ids[:, start:stop],
            position_ids=positions[:, start:stop],
            attention_mask=valid[:, start:stop],
            use_cache=True,
            past_key_values=cache,
        )
        cache = output.past_key_values
        assert cache.get_seq_length() == stop
        outputs.append(output.logits)
        start = stop
    torch.testing.assert_close(torch.cat(outputs, dim=1), expected, atol=1e-6, rtol=1e-5)
    cache.crop(-5)
    changed = ids.clone()
    changed[:, 7:] = 14
    expected = model(input_ids=changed, position_ids=positions, attention_mask=valid).logits[:, 7:]
    actual = model(
        input_ids=changed[:, 7:],
        position_ids=positions[:, 7:],
        attention_mask=valid[:, 7:],
        use_cache=True,
        past_key_values=cache,
    ).logits
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    cache.crop(-cache.get_seq_length())
    assert cache.get_seq_length() == 0
    actual = model(input_ids=ids[:, :3], use_cache=True, past_key_values=cache).logits
    torch.testing.assert_close(actual, model(input_ids=ids[:, :3]).logits, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("stride", [1, 2, 4])
@torch.no_grad()
def test_cached_isd_matches_prefix_recomputation(stride):
    torch.manual_seed(41)
    model = make_model().eval()
    model.lm_head.weight.mul_(15)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    results = [
        introspective_generate(
            model,
            ids,
            mask_token_id=31,
            max_new_tokens=32,
            stride=stride,
            generator=torch.Generator().manual_seed(42),
            use_cache=cache,
        )
        for cache in (False, True)
    ]
    expected, actual = results
    assert torch.equal(actual.sequences, expected.sequences)
    assert (actual.accepted, actual.proposed, actual.forward_passes) == (
        expected.accepted,
        expected.proposed,
        expected.forward_passes,
    )
    assert actual.processed_tokens < expected.processed_tokens
    if stride > 1:
        assert 0 < actual.accepted < actual.proposed


@torch.no_grad()
def test_cache_batch_selection_and_reset_preserve_metadata():
    model = make_model().eval()
    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    positions = torch.tensor([[0, 1, 2, 3], [0, 1, 0, 1]])
    valid = torch.tensor([[1, 0, 1, 1], [1, 1, 1, 1]])
    cache = model(input_ids=ids, position_ids=positions, attention_mask=valid, use_cache=True).past_key_values
    indices = torch.tensor([1, 0])
    cache.reorder_cache(indices)
    ids = ids[indices]
    positions, valid = positions[indices], valid[indices]
    cache.batch_repeat_interleave(2)
    ids = ids.repeat_interleave(2, dim=0)
    positions, valid = positions.repeat_interleave(2, dim=0), valid.repeat_interleave(2, dim=0)
    indices = torch.tensor([3, 0])
    cache.batch_select_indices(indices)
    ids = ids[indices]
    positions, valid = positions[indices], valid[indices]
    extension = torch.tensor([[9, 10], [11, 12]])
    extension_positions = positions[:, -1:] + torch.arange(1, 3)[None]
    full_valid = torch.cat((valid, torch.ones_like(extension)), dim=-1)
    actual = model(
        input_ids=extension,
        position_ids=extension_positions,
        attention_mask=full_valid,
        use_cache=True,
        past_key_values=cache,
    ).logits
    expected = model(
        input_ids=torch.cat((ids, extension), dim=-1),
        position_ids=torch.cat((positions, extension_positions), dim=-1),
        attention_mask=full_valid,
    ).logits[:, -2:]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    cache.reset()
    assert cache.get_seq_length() == 0
    actual = model(input_ids=ids[:1], use_cache=True, past_key_values=cache).logits
    torch.testing.assert_close(actual, model(input_ids=ids[:1]).logits, atol=1e-6, rtol=1e-5)


def test_cache_rejects_training_and_incompatible_cache():
    from transformers.cache_utils import DynamicCache

    model = make_model()
    ids = torch.tensor([[1, 2, 3]])
    with pytest.raises(ValueError, match="causal evaluation"):
        model(input_ids=ids, use_cache=True)
    model.eval()
    with pytest.raises(ValueError, match="Supervised"):
        model(input_ids=ids, labels=ids, use_cache=True)
    with pytest.raises(ValueError, match="Use MapleCache"):
        model(input_ids=ids, use_cache=True, past_key_values=DynamicCache())
    cache = model(input_ids=ids, use_cache=True).past_key_values
    with pytest.raises(ValueError, match="requires use_cache"):
        model(input_ids=ids, past_key_values=cache)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA cached FlexAttention parity")
@pytest.mark.parametrize("attention", ["flex_attention", "sdpa"])
@torch.no_grad()
def test_cuda_cached_attention_matches_full_forward(attention, cuda_parallel):
    torch.manual_seed(47)
    model = make_model(gpu=True, fused=True).eval()
    model.config._attn_implementation = attention
    ids = torch.randint(1, 30, (1, 48), device="cuda")
    positions = torch.cat((torch.arange(24), torch.arange(24)))[None].cuda()
    valid = torch.ones_like(ids)
    valid[:, 10] = 0
    expected = model(input_ids=ids, position_ids=positions, attention_mask=valid).logits
    cache = None
    outputs = []
    for start, stop in ((0, 17), (17, 30), (30, 48)):
        output = model(
            input_ids=ids[:, start:stop],
            position_ids=positions[:, start:stop],
            attention_mask=valid[:, start:stop],
            use_cache=True,
            past_key_values=cache,
        )
        cache = output.past_key_values
        outputs.append(output.logits)
    torch.testing.assert_close(torch.cat(outputs, dim=1), expected, atol=0.015, rtol=0.02)
    cache.crop(-18)
    changed = ids.clone()
    changed[:, 30:] = 14
    expected = model(input_ids=changed, position_ids=positions, attention_mask=valid).logits[:, 30:]
    actual = model(
        input_ids=changed[:, 30:],
        position_ids=positions[:, 30:],
        attention_mask=valid[:, 30:],
        use_cache=True,
        past_key_values=cache,
    ).logits
    torch.testing.assert_close(actual, expected, atol=0.015, rtol=0.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA cached ISD kernel integration")
def test_cuda_cached_isd_handles_single_token_and_speculative_suffixes(cuda_parallel):
    model = make_model(gpu=True, fused=True).eval()
    model.config._attn_implementation = "sdpa"
    prompt = torch.arange(1, 17, device="cuda")[None]
    for stride in (1, 2, 4):
        result = introspective_generate(
            model,
            prompt,
            mask_token_id=31,
            max_new_tokens=24,
            stride=stride,
            use_cache=True,
            generator=torch.Generator(device="cuda").manual_seed(17),
        )
        assert result.sequences.shape == (1, 40)
        assert result.processed_tokens <= prompt.numel() + (2 * stride - 1) * result.forward_passes


@pytest.fixture
def cuda_parallel(tmp_path):
    from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state

    owns_group = not torch.distributed.is_initialized()
    if owns_group:
        torch.distributed.init_process_group(
            backend="nccl", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1
        )
    _init_parallel_state()
    yield
    if owns_group:
        clear_parallel_state()
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel parity")
@pytest.mark.parametrize("block_size", [1, 3])
@pytest.mark.parametrize("attention", ["flex_attention", "sdpa"])
def test_cuda_training_kernels_match_reference(block_size, attention, cuda_parallel):
    torch.manual_seed(19)
    reference = make_model(gpu=True).train()
    reference.config.idlm_block_size = block_size
    reference_routes = []
    for layer in reference.model.layers:
        layer.mlp.gate.register_forward_hook(
            lambda module, inputs, output: reference_routes.append(output[0].detach().clone())
        )
    ids = torch.randint(1, 30, (1, 96), device="cuda")
    labels = ids.clone()
    labels[:, :7] = -100
    positions = torch.cat((torch.arange(48), torch.arange(48)))[None].cuda()
    valid = torch.ones_like(ids)
    valid[:, -5:] = 0
    labels[:, -5:] = -100
    # OpSlots are module-wide: finish the reference before binding fused ops.
    # The math backend is the independent reference, including fully padded rows.
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        expected_output = reference(input_ids=ids, labels=labels, position_ids=positions, attention_mask=valid)
        expected_output.loss.backward()
    actual = make_model(gpu=True, fused=True).train()
    actual.config._attn_implementation = attention
    actual.load_state_dict(reference.state_dict())
    actual.config.idlm_block_size = block_size
    for index, layer in enumerate(actual.model.layers):
        # Top-k membership is discontinuous at ties. Compare derivatives on the
        # same selected support, while retaining gradients through router scores.
        # The separate MoE test below checks the unmodified routing end to end.
        def same_support(module, inputs, output, index=index):
            indices = reference_routes[index]
            logits = output[2]
            weights = logits.softmax(-1).gather(-1, indices)
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
            return indices, weights, logits

        layer.mlp.gate.register_forward_hook(same_support)
    actual.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    output = actual(input_ids=ids, labels=labels, position_ids=positions, attention_mask=valid)
    output.loss.backward()
    torch.testing.assert_close(expected_output.loss, output.loss, atol=0.015, rtol=0.01)
    for (name, expected), (_, observed) in zip(reference.named_parameters(), actual.named_parameters()):
        assert expected.grad is not None and observed.grad is not None, name
        assert observed.grad.isfinite().all(), name
        # BF16 grouped GEMM moves route multiplication before the down projection.
        # Bound the whole-tensor gradient error, including expert/router gradients.
        error = (expected.grad.float() - observed.grad.float()).norm()
        scale = expected.grad.float().norm().clamp_min(1e-7)
        assert error / scale < 0.08, (name, float(error / scale))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA MoE parity")
def test_cuda_moe_forward_and_backward_with_native_routing(cuda_parallel):
    torch.manual_seed(17)
    reference = make_model(gpu=True).model.layers[0].mlp
    inputs = torch.randn(1, 96, 128, device="cuda", dtype=torch.bfloat16).requires_grad_()
    gradient = torch.randn_like(inputs)
    expected = reference(inputs)[0]
    expected.backward(gradient)
    actual = make_model(gpu=True, fused=True).model.layers[0].mlp
    actual.load_state_dict(reference.state_dict())
    actual_inputs = inputs.detach().clone().requires_grad_()
    output = actual(actual_inputs)[0]
    output.backward(gradient)
    for name, expected_tensor, observed in [
        ("output", expected, output),
        ("input gradient", inputs.grad, actual_inputs.grad),
        *[
            (name, param.grad, dict(actual.named_parameters())[name].grad)
            for name, param in reference.named_parameters()
        ],
    ]:
        assert expected_tensor is not None and observed is not None, name
        difference = (expected_tensor.float() - observed.float()).norm()
        assert difference / expected_tensor.float().norm().clamp_min(1e-7) < 0.04, name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FSDP2 CPU offload needs CUDA")
def test_single_gpu_offload_accumulates_the_reference_gradients(cuda_parallel, tmp_path):
    from veomni.arguments.arguments_types import MixedPrecisionConfig
    from veomni.distributed.torch_parallelize import build_parallelize_model
    from veomni.models.transformers.maple.runtime import initialize_mask_token

    torch.manual_seed(21)
    reference = make_model(gpu=True).train()
    reference.save_pretrained(tmp_path / "weights")
    actual = build_foundation_model(
        reference.config,
        torch_dtype="bfloat16",
        init_device="meta",
        ops_implementation=make_eager_ops_config(qat_implementation="ternary"),
    )
    actual = build_parallelize_model(
        actual,
        weights_path=str(tmp_path / "weights"),
        init_device="meta",
        enable_fsdp_offload=True,
        fsdp_offload_pin_memory=False,
        mixed_precision=MixedPrecisionConfig(enable=False),
        enable_gradient_checkpointing=True,
        basic_modules=["MapleDecoderLayer"],
    )
    assert next(actual.parameters()).device.type == "cpu"
    initialize_mask_token(reference, 31)
    initialize_mask_token(actual, 31)
    torch.testing.assert_close(reference.model.rotary_emb.inv_freq, actual.model.rotary_emb.inv_freq, atol=0, rtol=0)
    ids = torch.randint(1, 30, (1, 16), device="cuda")
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        for offset in (0, 1):
            expected = reference(input_ids=ids + offset, labels=ids + offset).loss / 2
            observed = actual(input_ids=ids + offset, labels=ids + offset).loss / 2
            torch.testing.assert_close(expected, observed, atol=0.002, rtol=0.002)
            expected.backward()
            observed.backward()
    for (name, expected), (_, observed) in zip(reference.named_parameters(), actual.named_parameters()):
        assert observed.grad is not None, name
        local = observed.grad.to_local().to("cuda")
        torch.testing.assert_close(expected.grad, local, atol=0.002, rtol=0.02, msg=name)


@pytest.mark.parametrize("block", [1, 2, 3])
@pytest.mark.parametrize("window", [None, 2])
def test_idlm_visibility_matches_independent_prefix_construction(block, window):
    positions = torch.tensor([[0, 1, 2, 3, 0, 1, 2]])
    valid = torch.tensor([[1, 1, 1, 1, 1, 1, 0]], dtype=torch.bool)
    actual = make_idlm_attention_mask(positions, valid, block, sliding_window=window)[0, 0]
    expected = torch.zeros(14, 14, dtype=torch.bool)
    for offset, length in [(0, 4), (4, 3)]:
        for i in range(length):
            if not valid[0, offset + i]:
                continue
            for j in range(i + 1):
                if not valid[0, offset + j] or (window is not None and i - j > window):
                    continue
                expected[7 + offset + i, 7 + offset + j] = True
                if j < (i // block) * block:
                    expected[offset + i, 7 + offset + j] = True
                else:
                    expected[offset + i, offset + j] = True
    assert torch.equal(actual, expected)


def test_shift_does_not_leak_across_documents_and_preserves_prompt():
    ids = torch.tensor([[10, 11, 12, 13, 20, 21]])
    labels = torch.tensor([[-100, -100, 12, 13, 20, 21]])
    pos = torch.tensor([[0, 1, 2, 3, 0, 1]])
    inputs, doubled_pos, noisy, clean = prepare_idlm_inputs(ids, labels, pos, 99)
    assert inputs.tolist() == [[10, 11, 99, 99, 99, 99, 10, 11, 12, 13, 20, 21]]
    assert noisy.tolist() == [[-100, -100, 13, -100, 21, -100]]
    assert clean.tolist() == [[-100, 12, 13, -100, 21, -100]]
    assert torch.equal(doubled_pos, pos.repeat(1, 2))


def test_auto_balance_does_not_differentiate_through_ratio():
    noisy, clean = torch.tensor(6.0, requires_grad=True), torch.tensor(2.0, requires_grad=True)
    balanced_idlm_loss(noisy, clean, auto_balance=True).backward()
    assert noisy.grad == 1
    assert clean.grad == 3


class LengthCache:
    def __init__(self):
        self.length = 0

    def get_seq_length(self):
        return self.length

    def crop(self, tokens_to_remove):
        assert tokens_to_remove < 0
        self.length = max(0, self.length + tokens_to_remove)


class ConstantModel(torch.nn.Module):
    def forward(self, input_ids, **kwargs):
        # Every position predicts token 1, including MASK positions.
        logits = torch.full((*input_ids.shape, 3), -1000.0)
        logits[..., 1] = 0
        cache = None
        if kwargs.get("use_cache"):
            cache = kwargs.get("past_key_values") or LengthCache()
            cache.length += input_ids.shape[-1]
        return SimpleNamespace(logits=logits, past_key_values=cache)


@pytest.mark.parametrize("stride", [1, 2, 4])
@pytest.mark.parametrize("limit", [0, 1, 2, 5, 13])
@pytest.mark.parametrize("use_cache", [False, True])
def test_isd_all_accept_length_and_fused_forwards(stride, limit, use_cache):
    result = introspective_generate(
        ConstantModel().eval(),
        torch.tensor([[0]]),
        mask_token_id=2,
        stride=stride,
        max_new_tokens=limit,
        use_cache=use_cache,
    )
    assert result.sequences.tolist() == [[0] + [1] * limit]
    assert result.accepted == result.proposed
    if stride == 4 and limit == 13:
        assert result.forward_passes == 4


@pytest.mark.parametrize("use_cache", [False, True])
def test_isd_rejects_invalid_drafts_and_handles_eos(use_cache):
    class OppositeDraft(ConstantModel):
        def forward(self, input_ids, **kwargs):
            result = super().forward(input_ids, **kwargs)
            result.logits[input_ids == 2] = torch.tensor([0.0, -1000.0, -1000.0])
            return result

    result = introspective_generate(
        OppositeDraft().eval(), torch.tensor([[0]]), mask_token_id=2, stride=4, max_new_tokens=9, use_cache=use_cache
    )
    assert result.sequences.tolist() == [[0] + [1] * 9]
    assert result.proposed > 0 and result.accepted == 0
    result = introspective_generate(
        ConstantModel().eval(), torch.tensor([[0]]), mask_token_id=2, eos_token_id=1, use_cache=use_cache
    )
    assert result.sequences.tolist() == [[0, 1]]


def test_isd_residual_correction_recovers_causal_sampling_distribution():
    target = torch.tensor([0.15, 0.65, 0.20])
    proposal = torch.tensor([0.70, 0.25, 0.05])

    class UnequalDraft(torch.nn.Module):
        def forward(self, input_ids, **kwargs):
            logits = target.log().expand(*input_ids.shape, -1).clone()
            logits[input_ids == 3] = proposal.log()
            return SimpleNamespace(logits=logits)

    generator = torch.Generator().manual_seed(123)
    model = UnequalDraft().eval()
    counts = torch.zeros(3)
    for _ in range(4000):
        result = introspective_generate(
            model, torch.tensor([[0]]), mask_token_id=3, stride=2, max_new_tokens=2, generator=generator
        )
        counts[result.sequences[0, -1]] += 1
    torch.testing.assert_close(counts / counts.sum(), target, atol=0.025, rtol=0)


def test_pretokenized_data_preserves_response_mask_and_leaves_shift_to_model():
    from veomni.data.data_transform import build_data_transform

    transform = build_data_transform("pretokenized", max_seq_len=3)
    record = transform(dict(input_ids=[1, 2, 3, 4], labels=[-100, -100, 3, 4]))[0]
    assert record["input_ids"].tolist() == [1, 2, 3]
    assert record["labels"].tolist() == [-100, -100, 3]
    assert record["attention_mask"].tolist() == [1, 1, 1]
    with pytest.raises(ValueError, match="aligned"):
        transform(dict(input_ids=[1, 2], labels=[1]))


@pytest.mark.parametrize("shape", [(3, 16), (4, 3, 16), (0, 16)])
def test_ternary_levels_and_identity_gradient(shape):
    weight = torch.randn(shape, requires_grad=True)
    output = ternary_fake_quant_weight(weight, 8)
    grouped = output.reshape(-1, 8)
    scale = grouped.abs().amax(-1, keepdim=True).clamp_min(1e-8)
    assert torch.isin(grouped / scale, torch.tensor([-1.0, 0.0, 1.0])).all()
    gradient = torch.randn_like(weight)
    output.backward(gradient)
    torch.testing.assert_close(weight.grad, gradient, rtol=0, atol=0)


def test_pretrained_ternary_is_not_rescaled():
    weight = torch.tensor([[0, 0, 0, 0.3, -0.3, 0, 0, 0], [0.1, -0.1, 0.1, 0, 0, 0.1, 0, 0]])
    torch.testing.assert_close(ternary_fake_quant_weight(weight, 8), weight, rtol=0, atol=0)
    assert torch.equal(ternary_fake_quant_weight(torch.zeros(2, 8), 8), torch.zeros(2, 8))


def test_qat_linear_updates_master_weights():
    inputs = torch.randn(5, 8, requires_grad=True)
    master = torch.randn(3, 8, requires_grad=True)
    quantized = ternary_fake_quant_weight(master, 4).detach()
    output = ternary_linear(inputs, master, group_size=4)
    expected = inputs.detach() @ quantized.T
    torch.testing.assert_close(output, expected)
    output.sum().backward()
    torch.testing.assert_close(inputs.grad, quantized.sum(0).expand_as(inputs))
    torch.testing.assert_close(master.grad, inputs.detach().sum(0).expand_as(master))


def test_bad_group_rejected():
    with pytest.raises(ValueError, match="divide"):
        ternary_fake_quant_weight(torch.randn(3, 9), 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA quantizer parity")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("groups,group_size", [(256, 128), (19, 128), (3, 8), (1, 2048), (0, 128)])
def test_cuda_quantizer_matches_cpu_and_ste(dtype, groups, group_size):
    torch.manual_seed(71)
    # The strided view checks noncontiguous inputs; 19 groups exercises a
    # partially filled final Triton program. Include zero and exact half ties.
    weight = torch.randn(groups, group_size * 2, dtype=dtype)[:, ::2]
    if groups:
        weight[0] = 0
        weight[-1, :4] = torch.tensor([4, 2, -2, -4], dtype=dtype)
    expected = ternary_fake_quant_weight(weight, group_size)
    actual_weight = torch.empty(groups, group_size * 2, dtype=dtype, device="cuda")[:, ::2]
    actual_weight.copy_(weight).requires_grad_()
    actual = ternary_fake_quant_weight(actual_weight, group_size)
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    torch.testing.assert_close(actual_weight.grad, gradient, atol=0, rtol=0)


@pytest.mark.parametrize("use_kahan", [True, False])
@pytest.mark.parametrize("multi", [True, False])
def test_anyprecision_resume_avoids_synthetic_gradient_step(tmp_path, use_kahan, multi):
    """Large CPU-offloaded resumes must not allocate a model-sized dummy gradient."""
    from veomni.checkpoint.dcp_checkpointer import ModelState, OptimizerState
    from veomni.optim.optimizer import AnyPrecisionAdamW, MultiOptimizer, initialize_optimizer_state_for_load

    parallel_state = SimpleNamespace(dp_mode="fsdp2")

    def optimizer_state(model, optimizer, load=False):
        wrapped = MultiOptimizer(model, {"adamw": optimizer}, ["adamw"]) if multi else optimizer
        return OptimizerState(model, wrapped, parallel_state=parallel_state, load=load)

    source = nn.Linear(8, 8).to(torch.bfloat16)
    source_optimizer = AnyPrecisionAdamW(source.parameters(), lr=0.01, weight_decay=0.1, use_kahan_summation=use_kahan)
    for param in source.parameters():
        param.grad = torch.randn_like(param)
    source_optimizer.step()
    source_optimizer.zero_grad(set_to_none=True)
    dcp.save(
        {
            "model": ModelState(source, parallel_state=parallel_state),
            "optimizer": optimizer_state(source, source_optimizer),
        },
        checkpoint_id=tmp_path,
    )

    target = nn.Linear(8, 8).to(torch.bfloat16)
    target_optimizer = AnyPrecisionAdamW(target.parameters(), lr=0.01, weight_decay=0.1, use_kahan_summation=use_kahan)
    with patch.object(target_optimizer, "step", side_effect=AssertionError("synthetic optimizer step")):
        dcp.load(
            {
                "model": ModelState(target, parallel_state=parallel_state),
                "optimizer": optimizer_state(target, target_optimizer, load=True),
            },
            checkpoint_id=tmp_path,
        )
    for source_param, target_param in zip(source.parameters(), target.parameters()):
        assert target_param.grad is None
        torch.testing.assert_close(target_param, source_param, rtol=0, atol=0)
        for key, value in source_optimizer.state[source_param].items():
            torch.testing.assert_close(target_optimizer.state[target_param][key], value, rtol=0, atol=0)
        source_param.grad = torch.randn_like(source_param)
        target_param.grad = source_param.grad.clone()
    initialize_optimizer_state_for_load(target_optimizer)
    initialize_optimizer_state_for_load(target_optimizer)
    source_optimizer.step()
    target_optimizer.step()
    for source_param, target_param in zip(source.parameters(), target.parameters()):
        torch.testing.assert_close(target_param, source_param, rtol=0, atol=0)
