"""Ternary weights with an identity straight-through estimator.

The detached absmax scale preserves a pretrained {-s, 0, s} group exactly.
In particular, applying an absmean scale to an already ternary checkpoint would
shrink its nonzero weights at the very first step. Master weights and optimizer
states remain floating point; this is training QAT, not packed integer inference.
Native Maple latent masters use row-wise TWN: threshold at 0.7 times mean
absolute weight, then scale by the mean absolute weight above that threshold.
"""

import torch
import torch.nn.functional as F


class _TernarySTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, group_size, scheme):
        if scheme not in ("group_absmax", "row_twn"):
            raise ValueError(f"Unknown ternary scheme: {scheme}")
        if scheme == "row_twn":
            from .ternary_twn import quantize_twn_native

            if weight.shape[-1] == 0:
                raise ValueError("TWN requires a nonempty input dimension")
            return quantize_twn_native(weight)
        if group_size <= 0 or weight.shape[-1] % group_size:
            raise ValueError("Ternary group_size must divide the weight's input dimension")
        if weight.is_cuda:
            from .ternary_triton import quantize_ternary_cuda

            return quantize_ternary_cuda(weight, group_size)
        groups = weight.float().reshape(-1, group_size)
        scale = groups.abs().amax(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
        quantized = (groups / scale).round().clamp(-1, 1) * scale
        return quantized.reshape_as(weight).to(weight.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def ternary_fake_quant_weight(weight: torch.Tensor, group_size: int = 128, *, scheme="group_absmax") -> torch.Tensor:
    """Apply group absmax or native Maple row-wise TWN, with identity STE."""
    return _TernarySTE.apply(weight, group_size, scheme)


def ternary_linear(inputs, weight, bias=None, *, group_size=128, scheme="group_absmax", enabled=True):
    if enabled:
        weight = ternary_fake_quant_weight(weight, group_size, scheme=scheme)
    return F.linear(inputs, weight, bias)
