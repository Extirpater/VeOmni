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


def _is_large_expert_stack(weight):
    return (
        weight.is_cuda
        and weight.dtype == torch.bfloat16
        and weight.ndim == 3
        and weight.is_contiguous()
        and weight.shape[-1] in (512, 2048)
        and weight.numel() >= 2**20
    )


@torch.library.custom_op("veomni::maple_twn_stats", mutates_args=())
def twn_statistics(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row threshold and scale of a large BF16 expert stack."""
    absolute = _absolute_values(weight)
    threshold = absolute.mean(-1, keepdim=True) * 0.7
    masked, count = _masked_values(absolute, threshold)
    del absolute
    alpha = masked.sum(-1, keepdim=True) / count.clamp_min(1.0)
    return threshold, alpha


@twn_statistics.register_fake
def _fake_statistics(weight):
    shape = (*weight.shape[:-1], 1)
    return weight.new_empty(shape, dtype=torch.float32), weight.new_empty(shape, dtype=torch.float32)


@torch.library.custom_op("veomni::maple_twn", mutates_args=())
def quantize_twn_native(weight: torch.Tensor) -> torch.Tensor:
    # Native reductions on strided inputs must retain their layout too. Compile
    # the large BF16 expert stacks; other shapes use the native expression.
    if not _is_large_expert_stack(weight):
        absolute = weight.float().abs()
        selected = absolute > absolute.mean(-1, keepdim=True) * 0.7
        scale = (absolute * selected).sum(-1, keepdim=True) / selected.sum(-1, keepdim=True).clamp_min(1)
        return (weight.float().sign() * selected * scale).to(weight.dtype)
    return _apply_values(weight, *twn_statistics(weight))


@quantize_twn_native.register_fake
def _fake(weight):
    return torch.empty_like(weight)


def quantize_twn_recompute_aware(weight, cache):
    """Quantize, reusing row statistics when activation checkpointing recomputes.

    A non-reentrant checkpoint recomputes inside backward with the same
    all-gathered weight as the forward that saved it, before any optimizer
    step. The last forward's statistics therefore describe that weight exactly,
    and only the pointwise apply runs again: the result is bitwise identical.
    ``cache`` is a per-parameter dict owned by the calling module.
    """
    if not _is_large_expert_stack(weight):
        return quantize_twn_native(weight)
    stats = cache.get("stats")
    recomputing = torch._C._current_graph_task_id() != -1
    if not recomputing or stats is None or stats[0].shape[:-1] != weight.shape[:-1]:
        stats = cache["stats"] = twn_statistics(weight)
    return _apply_values(weight, *stats)
