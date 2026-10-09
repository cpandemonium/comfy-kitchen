"""Isolated gfx906 research; never registered as a runtime backend."""
import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl


@g.jit
def cached_softmax(x_ptr, n_columns: gl.constexpr, scale: gl.constexpr):
    row = gl.program_id(0).to(gl.int64)
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [64, 1], [8, 1], [1, 0])
    thread = gl.arange(0, 512, layout=gl.SliceLayout(1, layout))
    slots0 = gl.arange(0, 64, layout=gl.SliceLayout(0, layout))
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


@g.jit
def tiled_pv(p_ptr, v_ptr, out_ptr, M: gl.constexpr, K: gl.constexpr,
             O_HEAD_STRIDE: gl.constexpr, BM: gl.constexpr = 32,
             BN: gl.constexpr = 64, BK: gl.constexpr = 32):
    head = gl.program_id(2).to(gl.int64)
    a_layout: gl.constexpr = gl.BlockedLayout([1, 4], [8, 8], [4, 1], [1, 0])
    b_layout: gl.constexpr = gl.BlockedLayout([1, 4], [8, 8], [1, 4], [1, 0])
    c_layout: gl.constexpr = gl.BlockedLayout([1, 4], [8, 8], [4, 1], [1, 0])
    m = gl.program_id(0) * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, a_layout))
    ka = gl.arange(0, BK, layout=gl.SliceLayout(0, a_layout))
    kb = gl.arange(0, BK, layout=gl.SliceLayout(1, b_layout))
    n = gl.program_id(1) * BN + gl.arange(0, BN, layout=gl.SliceLayout(0, b_layout))
    a_shared = gl.allocate_shared_memory(gl.float16, (BM, BK), gl.SwizzledSharedLayout(4, 1, 8, [1, 0]))
    b_shared = gl.allocate_shared_memory(gl.float16, (BK, BN), gl.SwizzledSharedLayout(4, 1, 8, [1, 0]))
    acc = gl.full((BM, BN), 0, gl.float32, c_layout)
    for start in range(triton.cdiv(K, BK)):
        a = gl.load(p_ptr + head * M * K + m[:, None].to(gl.int64) * K + start * BK + ka[None, :],
                    (m[:, None] < M) & (start * BK + ka[None, :] < K), other=0)
        b = gl.load(v_ptr + head * K * 128 + (start * BK + kb[:, None].to(gl.int64)) * 128 + n[None, :],
                    (start * BK + kb[:, None] < K) & (n[None, :] < 128), other=0)
        a_shared.store(a)
        b_shared.store(b)
        acc = gl.dot_fma(a_shared.load(gl.DotOperandLayout(0, c_layout, 0)),
                         b_shared.load(gl.DotOperandLayout(1, c_layout, 0)), acc)
    om = gl.program_id(0) * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, c_layout))
    on = gl.program_id(1) * BN + gl.arange(0, BN, layout=gl.SliceLayout(0, c_layout))
    gl.store(out_ptr + head * O_HEAD_STRIDE + om[:, None].to(gl.int64) * 128 + on[None, :], acc,
             (om[:, None] < M) & (on[None, :] < 128))
