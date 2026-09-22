"""Fused merged-expert activation, retaining the unfused dtype rounding points.

These helpers are internal to the Quack backend. Save the original projection
and the weighted output; recompute the cheap activation and clamp masks during
backward instead of retaining several full routed-token intermediates.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from ....utils.device import get_torch_device


@triton.jit
def _activation(gate, up, LIMIT: tl.constexpr, DTYPE: tl.constexpr):
    if LIMIT is not None:
        bound = tl.full((), LIMIT, tl.float32).to(DTYPE).to(tl.float32)
        gate = tl.minimum(gate, bound, propagate_nan=tl.PropagateNan.ALL)
        up = tl.minimum(
            tl.maximum(up, -bound, propagate_nan=tl.PropagateNan.ALL), bound, propagate_nan=tl.PropagateNan.ALL
        )
    silu = tl.div_rn(gate, 1.0 + libdevice.exp(-gate)).to(DTYPE).to(tl.float32)
    activation = (silu * up).to(DTYPE).to(tl.float32)
    return gate, up, silu, activation


@triton.jit
def _forward(
    PRE,
    ROUTE,
    OUT,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    LIMIT: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = tl.arange(0, BLOCK_C)
    mask = (rows[:, None] < ROWS) & (cols[None, :] < WIDTH)
    offsets = rows[:, None].to(tl.int64) * (2 * WIDTH) + cols[None, :]
    gate = tl.load(PRE + offsets, mask, 0).to(tl.float32)
    up = tl.load(PRE + offsets + WIDTH, mask, 0).to(tl.float32)
    route = tl.load(ROUTE + rows, rows < ROWS, 0).to(tl.float32)
    _, _, _, activation = _activation(gate, up, LIMIT, PRE.dtype.element_ty)
    tl.store(OUT + rows[:, None].to(tl.int64) * WIDTH + cols[None, :], activation * route[:, None], mask)


@triton.jit
def _backward(
    PRE,
    ROUTE,
    DY,
    DPRE,
    DROUTE,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    LIMIT: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = tl.arange(0, BLOCK_C)
    mask = (rows[:, None] < ROWS) & (cols[None, :] < WIDTH)
    offsets = rows[:, None].to(tl.int64) * (2 * WIDTH) + cols[None, :]
    raw_gate = tl.load(PRE + offsets, mask, 0).to(tl.float32)
    raw_up = tl.load(PRE + offsets + WIDTH, mask, 0).to(tl.float32)
    dy = tl.load(DY + rows[:, None].to(tl.int64) * WIDTH + cols[None, :], mask, 0).to(tl.float32)
    route = tl.load(ROUTE + rows, rows < ROWS, 0).to(tl.float32)
    dtype: tl.constexpr = PRE.dtype.element_ty
    gate, up, silu, activation = _activation(raw_gate, raw_up, LIMIT, dtype)
    # Each cast corresponds to a tensor materialized by the original PyTorch
    # sequence. Keeping them prevents a change to the low-precision operator.
    dactivation = (dy * route[:, None]).to(dtype).to(tl.float32)
    dup = (dactivation * silu).to(dtype).to(tl.float32)
    dsilu = (dactivation * up).to(dtype).to(tl.float32)
    sigmoid = tl.div_rn(1.0, 1.0 + libdevice.exp(-gate))
    dgate = ((dsilu * sigmoid) * tl.fma(gate, 1.0 - sigmoid, 1.0)).to(dtype).to(tl.float32)
    if LIMIT is not None:
        bound = tl.full((), LIMIT, tl.float32).to(dtype).to(tl.float32)
        dgate = tl.where(raw_gate <= bound, dgate, 0.0)
        dup = tl.where((raw_up >= -bound) & (raw_up <= bound), dup, 0.0)
    tl.store(DPRE + offsets, dgate, mask)
    tl.store(DPRE + offsets + WIDTH, dup, mask)
    products = (activation * dy).to(dtype).to(tl.float32)
    droute = tl.sum(tl.where(cols[None, :] < WIDTH, products, 0.0), axis=1)
    tl.store(DROUTE + rows, droute, rows < ROWS)


def _launch_options(width):
    # Several rows per CTA amortize scheduling overhead on million-row expert
    # projections while retaining a complete row for the routing reduction.
    block_c = triton.next_power_of_2(width)
    return dict(BLOCK_R=max(1, min(8, 2048 // block_c)), BLOCK_C=block_c, num_warps=4, enable_fp_fusion=False)


def weighted_swiglu_forward(preactivation, routing_weights, limit):
    rows, doubled_width = preactivation.shape
    width = doubled_width // 2
    output = torch.empty((rows, width), device=preactivation.device, dtype=preactivation.dtype)
    if rows:
        options = _launch_options(width)
        with get_torch_device().device(preactivation.device):
            _forward[(triton.cdiv(rows, options["BLOCK_R"]),)](
                preactivation, routing_weights, output, rows, width, limit, **options
            )
    return output


def weighted_swiglu_backward(preactivation, routing_weights, grad_output, limit):
    rows, doubled_width = preactivation.shape
    width = doubled_width // 2
    grad_preactivation = torch.empty_like(preactivation)
    grad_routing = torch.empty_like(routing_weights)
    if rows:
        options = _launch_options(width)
        with get_torch_device().device(preactivation.device):
            _backward[(triton.cdiv(rows, options["BLOCK_R"]),)](
                preactivation,
                routing_weights,
                grad_output,
                grad_preactivation,
                grad_routing,
                rows,
                width,
                limit,
                **options,
            )
    return grad_preactivation, grad_routing
