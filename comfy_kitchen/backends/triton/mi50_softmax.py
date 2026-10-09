"""Experimental FP16 scale/softmax matching ROCm's 512-thread reduction order."""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl


@triton.jit
def _scale_softmax(x_ptr, n_columns: tl.constexpr, scale: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    threads = tl.arange(0, 512)
    lane = tl.arange(0, 8)
    columns = threads[:, None] * 8 + lane[None, :]
    maximum = tl.full((512,), float("-inf"), tl.float32)
    for chunk in range(triton.cdiv(n_columns, 4096)):
        indices = columns + chunk * 4096
        values = tl.load(x_ptr + row * n_columns + indices, indices < n_columns, other=0).to(
            tl.float32
        )
        values = (values * scale).to(tl.float16).to(tl.float32)
        maximum = tl.maximum(
            maximum, tl.max(tl.where(indices < n_columns, values, float("-inf")), 1)
        )
    maximum = tl.max(maximum, 0)
    partial = tl.full((512,), 0, tl.float32)
    for chunk in range(triton.cdiv(n_columns, 4096)):
        indices = columns + chunk * 4096
        values = tl.load(x_ptr + row * n_columns + indices, indices < n_columns, other=0).to(
            tl.float32
        )
        values = (values * scale).to(tl.float16).to(tl.float32)
        numerator = tl.where(indices < n_columns, tl.exp(values - maximum), 0)
        for j in tl.static_range(8):
            partial += tl.gather(numerator, tl.full((512, 1), j, tl.int32), 1).reshape(512)
    # ROCm BlockReduce: one Wave64 sum, followed by the eight wave sums.
    waves = partial.reshape(8, 64)
    wave_lane = tl.arange(0, 64)
    for shift in tl.static_range(6):
        offset = 32 >> shift
        source = tl.where(wave_lane + offset < 64, wave_lane + offset, wave_lane)
        partner = tl.gather(waves, tl.broadcast_to(source[None, :], (8, 64)), 1)
        waves = waves + partner
    wave_sums = tl.gather(waves, tl.full((8, 1), 0, tl.int32), 1).reshape(8)
    wave = tl.arange(0, 8)
    for shift in tl.static_range(3):
        offset = 4 >> shift
        source = tl.where(wave + offset < 8, wave + offset, wave)
        wave_sums = wave_sums + tl.gather(wave_sums, source, 0)
    total = tl.sum(tl.where(wave == 0, wave_sums, 0), 0)
    inverse = tl.div_rn(1.0, total)
    for chunk in range(triton.cdiv(n_columns, 4096)):
        indices = columns + chunk * 4096
        values = tl.load(x_ptr + row * n_columns + indices, indices < n_columns, other=0).to(
            tl.float32
        )
        values = (values * scale).to(tl.float16).to(tl.float32)
        result = tl.exp(values - maximum) * inverse
        tl.store(x_ptr + row * n_columns + indices, result, indices < n_columns)


@g.jit
def _cached_scale_softmax(x_ptr, n_columns: gl.constexpr, scale: gl.constexpr):
    row = gl.program_id(0).to(gl.int64)
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [64, 1], [8, 1], [1, 0])
    thread = gl.arange(0, 512, layout=gl.SliceLayout(1, layout))
    cache_width: gl.constexpr = 32 if n_columns == 10120 else 64
    slots0 = gl.arange(0, cache_width, layout=gl.SliceLayout(0, layout))
    slots1 = gl.arange(0, 16, layout=gl.SliceLayout(0, layout)) + 64
    idx0 = thread[:, None] * 8 + slots0[None, :] % 8 + slots0[None, :] // 8 * 4096
    idx1 = thread[:, None] * 8 + slots1[None, :] % 8 + slots1[None, :] // 8 * 4096
    cache0 = (gl.load(x_ptr + row * n_columns + idx0, idx0 < n_columns, other=0).to(gl.float32) * scale).to(gl.float16)
    cache1 = (gl.load(x_ptr + row * n_columns + idx1, idx1 < n_columns, other=0).to(gl.float32) * scale).to(gl.float16)
    maximum = gl.maximum(gl.max(gl.where(idx0 < n_columns, cache0.to(gl.float32), float('-inf')), 1),
                         gl.max(gl.where(idx1 < n_columns, cache1.to(gl.float32), float('-inf')), 1))
    maximum = gl.max(maximum, 0)
    partial = gl.full((512,), 0, gl.float32, gl.SliceLayout(1, layout))
    for j in gl.static_range(triton.cdiv(n_columns, 4096) * 8):
        if j < 64:
            value = gl.gather(cache0, gl.full((512, 1), j, gl.int32, layout), 1).reshape((512,)).to(gl.float32)
        else:
            value = gl.gather(cache1, gl.full((512, 1), j - 64, gl.int32, layout), 1).reshape((512,)).to(gl.float32)
        value = gl.convert_layout(value, gl.SliceLayout(1, layout))
        valid = thread * 8 + j % 8 + j // 8 * 4096 < n_columns
        partial = partial + gl.where(valid, gl.exp(value - maximum), 0)
    # Same serial per-thread accumulation and descending Wave64 reduction as ROCm.
    wave_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [8, 1], [1, 0])
    waves = gl.convert_layout(partial.reshape((8, 64)), wave_layout)
    lane = gl.arange(0, 64, layout=gl.SliceLayout(0, wave_layout))
    for shift in gl.static_range(6):
        offset = 32 >> shift
        source = gl.where(lane + offset < 64, lane + offset, lane)
        indices = gl.full((8, 1), 0, gl.int32, wave_layout) + source[None, :]
        waves = waves + gl.gather(waves, indices, 1)
    sums = gl.gather(waves, gl.full((8, 1), 0, gl.int32, wave_layout), 1).reshape((8,))
    wave = gl.arange(0, 8, layout=gl.SliceLayout(1, wave_layout))
    sums = gl.convert_layout(sums, gl.SliceLayout(1, wave_layout))
    for shift in gl.static_range(3):
        offset = 4 >> shift
        source = gl.where(wave + offset < 8, wave + offset, wave)
        sums = sums + gl.gather(sums, source, 0)
    total = gl.sum(gl.where(wave == 0, sums, 0), 0)
    inverse = gl.div_rn(1.0, total)
    gl.store(x_ptr + row * n_columns + idx0, gl.exp(cache0.to(gl.float32) - maximum) * inverse, idx0 < n_columns)
    gl.store(x_ptr + row * n_columns + idx1, gl.exp(cache1.to(gl.float32) - maximum) * inverse, idx1 < n_columns)


def scale_softmax_inplace(scores, scale):
    """Exact cached softmax for the three validated LTX gfx906 row lengths.

    Rounded FP16 scores stay in registers; exp and ordered FP32 accumulation
    preserve ROCm rounding. No extra score buffer or global scratch is used.
    The original three-read kernel remains available for research/rollback.
    """
    if (
        torch.is_grad_enabled()
        or not torch.version.hip
        or scores.device.type != "cuda"
        or scores.dtype != torch.float16
        or scores.ndim < 2
        or scores.shape[-1] not in (5824, 10120, 40480)
        or not scores.is_contiguous()
        or not torch.cuda.get_device_properties(scores.device).gcnArchName.startswith("gfx906")
    ):
        raise ValueError(
            "scale_softmax_inplace requires contiguous 5824/10120/40480-wide FP16 gfx906 inference scores"
        )
    _cached_scale_softmax[(scores.numel() // scores.shape[-1],)](
        scores, scores.shape[-1], scale, num_warps=8, enable_fp_fusion=False
    )
    return scores
