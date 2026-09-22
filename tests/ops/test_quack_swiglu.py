"""Activation parity on CUDA, including GPUs below Quack GEMM's SM90 minimum."""

import pytest
import torch
import torch.nn.functional as F


pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("limit", [None, 7.0, 0.3])
@pytest.mark.parametrize("rows,width", [(0, 64), (37, 67), (259, 512), (7, 2048)])
def test_weighted_swiglu_preserves_rounding_and_clamp_gradients(dtype, limit, rows, width):
    from veomni.ops.kernels.moe._quack_swiglu import weighted_swiglu_backward, weighted_swiglu_forward

    torch.manual_seed(107)
    preactivation = (torch.randn(rows, 2 * width, device="cuda", dtype=dtype) * 8).requires_grad_()
    route = torch.rand(rows, 1, device="cuda", dtype=dtype, requires_grad=True)
    if rows and limit is not None:
        with torch.no_grad():
            # The boundary itself has a nonzero clamp derivative. Also cover
            # a scalar limit that is not exactly representable in the dtype.
            boundary = torch.tensor(limit, device="cuda", dtype=dtype)
            preactivation[0, :3] = torch.stack((boundary, -boundary, boundary + 1))
            preactivation[0, width : width + 3] = torch.stack((boundary, -boundary, -boundary - 1))
    gate, up = preactivation.chunk(2, dim=-1)
    if limit is not None:
        gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
    expected = (F.silu(gate) * up) * route
    grad_output = torch.randn_like(expected)
    expected.backward(grad_output)
    actual = weighted_swiglu_forward(preactivation.detach(), route.detach(), limit)
    grad_pre, grad_route = weighted_swiglu_backward(preactivation.detach(), route.detach(), grad_output, limit)
    tolerance = 0.008 if dtype == torch.bfloat16 else 0.001
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=tolerance)
    torch.testing.assert_close(grad_pre, preactivation.grad, atol=2e-6, rtol=tolerance)
    torch.testing.assert_close(grad_route, route.grad, atol=2e-6, rtol=tolerance)
