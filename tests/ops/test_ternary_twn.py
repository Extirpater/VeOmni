"""Native Maple TWN numerics and identity-STE gradients on CPU and CUDA."""

import pytest
import torch

from veomni.ops.qat.ternary import ternary_fake_quant_weight


def reference_twn(weight):
    # Native checkpoint recipe: threshold and scale independently per output row.
    values = weight.detach().float()
    absolute = values.abs()
    mask = absolute > 0.7 * absolute.mean(-1, keepdim=True)
    alpha = (absolute * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True).clamp_min(1)
    return (values.sign() * mask * alpha).to(weight.dtype)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(3, 5), (2, 3, 128), (3, 2048), (3, 512, 2048), (3, 2048, 512), (0, 16)])
def test_native_twn_forward_and_identity_gradient(device, dtype, shape):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(18)
    # Transpose twice through a contiguous copy to retain a noncontiguous view.
    weight = torch.randn(shape).transpose(-1, -2).contiguous().transpose(-1, -2).to(dtype).requires_grad_()
    actual_weight = weight.detach().to(device).requires_grad_()
    expected = reference_twn(actual_weight)
    actual = ternary_fake_quant_weight(actual_weight, scheme="row_twn")
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    torch.testing.assert_close(actual_weight.grad, gradient, atol=0, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_native_twn_zero_rows_and_strict_threshold(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    weight = torch.tensor([[0.0, 0.0], [0.7, 1.3], [-0.7, -1.3]], device=device)
    actual = ternary_fake_quant_weight(weight, scheme="row_twn")
    expected = torch.tensor([[0.0, 0.0], [0.0, 1.3], [0.0, -1.3]], device=device)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA expert-stack path")
@pytest.mark.parametrize("rows,width", [(512, 2048), (2048, 512)])
def test_large_contiguous_experts_preserve_native_reductions(rows, width):
    torch.manual_seed(97)
    weight = torch.randn(3, rows, width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    expected = reference_twn(weight)
    actual = ternary_fake_quant_weight(weight, scheme="row_twn")
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    torch.testing.assert_close(weight.grad, gradient, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA expert-stack path")
@pytest.mark.parametrize("scheme", ["row_twn", "group_absmax"])
def test_merging_gate_up_preserves_ternary_states(scheme):
    torch.manual_seed(101)
    gate = torch.randn(3, 512, 2048, device="cuda", dtype=torch.bfloat16)
    up = torch.randn_like(gate)
    expected = torch.cat([ternary_fake_quant_weight(weight, scheme=scheme) for weight in (gate, up)], dim=1)
    merged = ternary_fake_quant_weight(torch.cat((gate, up), dim=1), scheme=scheme)
    torch.testing.assert_close(merged, expected, atol=0, rtol=0)
