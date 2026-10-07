"""Experimental FP16 scale/softmax matching ROCm's 512-thread reduction order."""

import torch
import triton
import triton.language as tl


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


def scale_softmax_inplace(scores, scale):
    """Caller supplies contiguous 40480-wide FP16 score rows on gfx906."""
    if (
        torch.is_grad_enabled()
        or not torch.version.hip
        or scores.device.type != "cuda"
        or scores.dtype != torch.float16
        or scores.ndim < 2
        or scores.shape[-1] != 40480
        or not scores.is_contiguous()
        or not torch.cuda.get_device_properties(scores.device).gcnArchName.startswith("gfx906")
    ):
        raise ValueError(
            "scale_softmax_inplace requires contiguous 40480-wide FP16 gfx906 inference scores"
        )
    _scale_softmax[(scores.numel() // scores.shape[-1],)](
        scores, scores.shape[-1], scale, num_warps=8, enable_fp_fusion=False
    )
    return scores
