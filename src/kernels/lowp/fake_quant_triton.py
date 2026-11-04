"""Fused Triton MXFP8 fake quantization: one kernel for the quantize-dequantize round trip that
:func:`~src.kernels.lowp.quantization.fake_quant` takes on CUDA, contiguous, last-axis tensors.

Each program handles ``BPP`` blocks of ``block_size`` elements: it reduces the block amax, picks the
``e8m0`` exponent, rounds to ``e4m3`` and rescales, in registers, so the tensor is read and written once.
Output is bit-identical to the eager :func:`~src.kernels.lowp.quantization._block_round_trip`, NaN
blocks and subnormal scales included; the GPU test pins that on adversarial inputs.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_TINY = torch.finfo(torch.float32).tiny  # 2^-126


@triton.jit
def _mul_rn(a, b):
    """IEEE fp32 multiply with subnormals preserved. Triton compiles arithmetic/libdevice with
    flush-to-zero; PyTorch's CUDA kernels don't, so the scaling path must be explicit PTX."""
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _pow2(e):
    """int32 e in [-127, 127] -> fp32 2^e, built from bits so 2^-127 (subnormal) is exact."""
    bits = tl.where(e == -127, 0x00400000, (e + 127) << 23)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _mxfp8_fake_quant_kernel(x_ptr, out_ptr, n_blocks, BPP: tl.constexpr, BLOCK: tl.constexpr, TINY: tl.constexpr):
    blk = tl.program_id(0).to(tl.int64) * BPP + tl.arange(0, BPP)
    lane = tl.arange(0, BLOCK)
    offs = blk[:, None] * BLOCK + lane[None, :]
    mask = (blk < n_blocks)[:, None] & (lane < BLOCK)[None, :]
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    has_nan = tl.sum((x != x).to(tl.int32), axis=1) > 0
    amax = tl.max(tl.abs(x), axis=1)
    y = tl.math.div_rn(tl.maximum(amax, TINY), 448.0)
    se = libdevice.ceil(libdevice.log2(y))
    se = tl.minimum(tl.maximum(se, -127.0), 127.0)
    se_i = se.to(tl.int32)

    v = _mul_rn(x, _pow2(-se_i)[:, None])
    v = tl.minimum(tl.maximum(v, -448.0), 448.0)
    q = v.to(tl.float8e4nv).to(tl.float32)
    out = _mul_rn(q, _pow2(se_i)[:, None])
    out = tl.where(has_nan[:, None], float("nan"), out)
    tl.store(out_ptr + offs, out.to(tl.bfloat16).to(out_ptr.dtype.element_ty), mask=mask)


def fake_quant_mxfp8_triton(t: torch.Tensor, block_size: int, BPP: int = 64) -> torch.Tensor:
    """Last-axis, contiguous, CUDA. Caller guarantees ``t.shape[-1] % block_size == 0``."""
    if t.numel() == 0:
        return torch.empty_like(t)
    out = torch.empty_like(t)
    n_blocks = t.numel() // block_size
    _mxfp8_fake_quant_kernel[(triton.cdiv(n_blocks, BPP),)](t, out, n_blocks, BPP=BPP, BLOCK=block_size, TINY=_TINY)
    return out
