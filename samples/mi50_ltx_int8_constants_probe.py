"""Research: LTX FFN constants, bounded row tail, unchanged INT32 accumulation."""
import triton
import triton.language as tl

@triton.jit
def _ltx_int8_constants(
    # Pointers
    a_ptr, b_ptr, c_ptr,
    a_scale_ptr, b_scale_ptr, bias_ptr,
    # Matrix Dimensions
    m, n: tl.constexpr, k: tl.constexpr,
    # Strides
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    # Meta-parameters
    block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr,
    group_size_m: tl.constexpr,
    has_bias: tl.constexpr
):
    """
    Computes: C = ((A * B) * (scale_a[:, None] * scale_b[None, :])) + bias
    A: [m, k] int8, scale_a: [m, 1] per-row activation scales
    B: [n, k] int8, scale_b: [n, 1] per-row weight scales
    """
    tl.static_assert(k % block_k == 0 and n % block_n == 0)
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(m, block_m)
    num_pid_n = tl.cdiv(n, block_n)
    num_pid_in_group = group_size_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * group_size_m
    actual_group_size_m = min(num_pid_m - first_pid_m, group_size_m)
    pid_m = first_pid_m + (pid % actual_group_size_m)
    pid_n = (pid % num_pid_in_group) // actual_group_size_m

    # 1. Prepare Pointers for A and B
    offs_am = pid_m * block_m + tl.arange(0, block_m)
    offs_bn = pid_n * block_n + tl.arange(0, block_n)
    offs_k = tl.arange(0, block_k)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # 2. Main Loop (Accumulate in Int32)
    accumulator = tl.zeros((block_m, block_n), dtype=tl.int32)

    for k_idx in range(0, tl.cdiv(k, block_k)):
        a = tl.load(a_ptrs, mask=offs_am[:, None] < m, other=0)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += block_k * stride_ak
        b_ptrs += block_k * stride_bk

    # 3. Fused Epilogue (Dequantize & Bias)
    scale_a = tl.load(a_scale_ptr + offs_am, mask=offs_am < m, other=0)  # Vector [BLOCK_M]
    scale_b = tl.load(b_scale_ptr + offs_bn)  # Vector [BLOCK_N]

    c = accumulator.to(tl.float32)
    total_scale = scale_a[:, None] * scale_b[None, :]
    c = c * total_scale

    if has_bias:
        bias = tl.load(bias_ptr + offs_bn)
        c = c + bias[None, :]

    # 4. Store Result
    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    c_mask = (offs_am[:, None] < m) & (offs_bn[None, :] < n)
    tl.store(c_ptrs, c, mask=c_mask)
