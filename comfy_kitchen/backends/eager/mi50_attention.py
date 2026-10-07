"""Dense long-video attention for the validated MI50 LTX refiner shape.

All keys participate in every softmax. FP16 rounding can differ slightly from
PyTorch softmax because reduction trees and exp implementations differ.
Triton is used only for a Wave64 scale/softmax kernel, not matrix products.
"""
from functools import lru_cache
import os

import torch

DEFAULT_VARIANT = dict(
    head_chunk=8,
    query_chunk=4096,
    contiguous_k=False,
    fused_softmax=os.environ.get("MI50_LTX_FUSED_SOFTMAX", "0") == "1",
    inplace_softmax=True,
    ordered_softmax=False,
)


def can_use(q, k, v, heads, mask=None, attn_precision=None, **kwargs):
    return (
        torch.version.hip is not None
        and not torch.is_grad_enabled()
        and q.device.type == "cuda"
        and q.dtype == k.dtype == v.dtype == torch.float16
        and q.device == k.device == v.device
        and q.ndim == k.ndim == v.ndim == 3
        and q.shape == k.shape == v.shape == (1, 40480, 4096)
        and heads == 32 and mask is None
        and attn_precision != torch.float32
        and not kwargs.get("skip_reshape", False)
        and not kwargs.get("enable_gqa", False)
        and not kwargs.get("is_causal", False)
        and kwargs.get("dropout_p", 0.) == 0.
        and torch.cuda.get_device_properties(q.device).gcnArchName.startswith("gfx906")
    )


@lru_cache(maxsize=1)
def _softmax_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(X, Y, N: tl.constexpr, SCALE: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        columns = tl.arange(0, BLOCK)
        in_row = columns < N
        # Match PyTorch: fp16 scale, then fp32 max/exp/sum. Mask before the
        # reduction so padding cannot change the row max or the partition.
        values = tl.load(X + row * N + columns, mask=in_row, other=0).to(tl.float32)
        values = (values * SCALE).to(tl.float16).to(tl.float32)
        values = tl.where(in_row, values, float("-inf"))
        shifted = values - tl.max(values, 0)
        numerator = tl.where(in_row, tl.exp(shifted), 0.0)
        result = numerator / tl.sum(numerator, 0)
        tl.store(Y + row * N + columns, result, mask=in_row)
    return kernel


def long_attention(q, k, v, heads=32, scale=None, max_score_bytes=2**32,
                   fused_softmax=False, contiguous_k=False, head_chunk=8, query_chunk=4096,
                   inplace_softmax=True, ordered_softmax=False):
    """Validated dense FP16 inference; caller must check can_use first.

    max_score_bytes bounds one score/probability buffer, not total live memory.
    In-place PyTorch softmax reuses the score storage and retains its reduction
    algorithm. The experimental fused path still needs two buffers. Callers
    must also account for packed Q/K/V and output when sizing the budget.
    """
    if not can_use(q, k, v, heads):
        raise ValueError("MI50 long_attention requires the validated gfx906 FP16 inference shape")
    if fused_softmax and ordered_softmax:
        raise ValueError("fused_softmax and ordered_softmax are mutually exclusive")
    if max_score_bytes < 40480 * 2:
        raise ValueError("max_score_bytes cannot hold even one attention row")
    b, n, c = q.shape
    d = c // heads
    scale = d ** -.5 if scale is None else scale
    q, k, v = [t.reshape(b, n, heads, d).permute(0, 2, 1, 3)
               .reshape(b*heads, n, d).contiguous() for t in (q, k, v)]
    kt = k.transpose(-2, -1)
    if contiguous_k:
        kt = kt.contiguous()
    output = torch.empty_like(q)
    # Leave the model manager's reserved memory intact. Drop head/query batch
    # sizes together if less temporary memory is available than in validation.
    while head_chunk > 1 and head_chunk * 1024 * n * 2 > max_score_bytes:
        head_chunk //= 2
    query_chunk = min(query_chunk, max(1, max_score_bytes // (head_chunk * n * 2)))
    kernel = _softmax_kernel() if fused_softmax else None
    for h in range(0, b*heads, head_chunk):
        for start in range(0, n, query_chunk):
            qs = q[h:h+head_chunk, start:start+query_chunk]
            scores = torch.bmm(qs, kt[h:h+head_chunk])
            if ordered_softmax:
                from comfy_kitchen.backends.triton.mi50_softmax import scale_softmax_inplace
                probabilities = scale_softmax_inplace(scores, scale)
            elif fused_softmax:
                probabilities = torch.empty_like(scores)
                kernel[(scores.numel()//n,)](scores, probabilities, n, scale, 65536, num_warps=8)
            else:
                scores.mul_(scale)
                if inplace_softmax:
                    probabilities = torch.softmax(scores, dim=-1, out=scores)
                else:
                    probabilities = scores.softmax(-1)
            del scores
            torch.bmm(probabilities, v[h:h+head_chunk],
                      out=output[h:h+head_chunk, start:start+query_chunk])
            del probabilities
    return output.reshape(b, heads, n, d).permute(0, 2, 1, 3).reshape(b, n, c)
