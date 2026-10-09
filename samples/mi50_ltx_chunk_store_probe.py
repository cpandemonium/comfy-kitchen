"""Research: ordered FP16 cache, eight-value FP32 exp/store temporaries."""
import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

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
    lanes = gl.arange(0, 8, layout=gl.SliceLayout(0, layout))
    for block in gl.static_range(triton.cdiv(n_columns, 4096)):
        gather_idx = gl.full((512, 1), block * 8 if block < 8 else (block - 8) * 8, gl.int32, layout) + lanes[None, :]
        if block < 8:
            values = gl.gather(cache0, gather_idx, 1).to(gl.float32)
        else:
            values = gl.gather(cache1, gather_idx, 1).to(gl.float32)
        indices = thread[:, None] * 8 + lanes[None, :] + block * 4096
        probability = gl.exp(values - maximum) * inverse
        gl.store(x_ptr + row * n_columns + indices, probability, indices < n_columns)

