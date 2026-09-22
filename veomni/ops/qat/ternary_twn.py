"""Native Maple TWN with unchanged PyTorch floating-point reductions.

Changing reduction order can move a weight across the ternary threshold.
Compile only pointwise work and the exactly representable selected count.
The custom-op boundary keeps enclosing compilation from fusing the reductions.
"""

import torch


@torch.compile(fullgraph=True, dynamic=True)
def _absolute_values(weight):
    return weight.float().abs()


@torch.compile(fullgraph=True, dynamic=True)
def _masked_values(absolute, threshold):
    mask = (absolute > threshold).float()
    return absolute * mask, mask.sum(-1, keepdim=True)


@torch.compile(fullgraph=True, dynamic=True)
def _apply_values(weight, threshold, alpha):
    values = weight.float()
    return (values.sign() * (values.abs() > threshold).float() * alpha).to(weight.dtype)


@torch.library.custom_op("veomni::maple_twn", mutates_args=())
def quantize_twn_native(weight: torch.Tensor) -> torch.Tensor:
    # Native reductions on strided inputs must retain their layout too. Compile
    # the large BF16 expert stacks; other shapes use the native expression.
    large_experts = (
        weight.is_cuda
        and weight.dtype == torch.bfloat16
        and weight.ndim == 3
        and weight.is_contiguous()
        and weight.shape[-1] in (512, 2048)
        and weight.numel() >= 2**20
    )
    if not large_experts:
        absolute = weight.float().abs()
        selected = absolute > absolute.mean(-1, keepdim=True) * 0.7
        scale = (absolute * selected).sum(-1, keepdim=True) / selected.sum(-1, keepdim=True).clamp_min(1)
        return (weight.float().sign() * selected * scale).to(weight.dtype)
    absolute = _absolute_values(weight)
    threshold = absolute.mean(-1, keepdim=True) * 0.7
    masked, count = _masked_values(absolute, threshold)
    del absolute
    alpha = masked.sum(-1, keepdim=True) / count.clamp_min(1.0)
    del masked
    return _apply_values(weight, threshold, alpha)


@quantize_twn_native.register_fake
def _fake(weight):
    return torch.empty_like(weight)
