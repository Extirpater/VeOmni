"""CUDA ternary forward quantizer; its STE backward is owned by ternary.py."""

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize(weight, output, num_groups, GROUP: tl.constexpr, ROWS: tl.constexpr):
    groups = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    indices = groups[:, None] * GROUP + tl.arange(0, GROUP)[None, :]
    valid = groups[:, None] < num_groups
    values = tl.load(weight + indices, valid, other=0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(values), 1), 1.1754943508222875e-38)[:, None]
    normalized = tl.div_rn(values, scale)
    # Absmax normalization is in [-1, 1]; ties at +/-0.5 round to zero.
    quantized = tl.where(tl.abs(normalized) > 0.5, tl.where(values > 0, scale, -scale), 0.0)
    tl.store(output + indices, quantized, valid)


def quantize_ternary_cuda(weight, group_size):
    if group_size & (group_size - 1):
        raise ValueError("The Triton ternary kernel requires a power-of-two group size")
    contiguous = weight.contiguous()
    output = torch.empty_like(contiguous)
    num_groups = weight.numel() // group_size
    if num_groups:
        # Several independent reductions share a program instead of launching
        # one mostly idle program for every 128-weight group.
        rows = min(8, max(1, 1024 // group_size))
        _quantize[(triton.cdiv(num_groups, rows),)](contiguous, output, num_groups, GROUP=group_size, ROWS=rows)
    return output
