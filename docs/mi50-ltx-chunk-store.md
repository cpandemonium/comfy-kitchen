# LTX N40480: eight-value ordered softmax stores

On gfx906, 2026-10-09, the existing FP16 score cache and ordered FP32
denominator are retained. Final probabilities are calculated/stored eight
values at a time. All keys and the original reduction order remain intact.
The compiler reuses exponentials from denominator calculation: v_exp/v_ldexp
instructions fall from 160/160 to 80/80, total instructions 2120 to 1421.

Compilation initially failed a strict zero-scratch/register-reduction gate:
VGPR125/private0/spill0 becomes VGPR128/private20/spill4, with six folded
reloads. A separate tradeoff assessment admitted the same candidate for one
numerical/timing pair because it halves exponentials with bounded scratch.
No second softmax candidate was generated. This preserves the rejected
static assessment rather than treating a timing result as compile evidence.

Captured real N40480 Q/K/V, all 32 heads, 128 dimensions, FP16 were used.
32-query and 17-query-tail softmax tests include zero, constant and extreme
finite rows. Complete attention and softmax output are bitwise equal and
finite, with zero differing FP16 values.

One warmed pair, scores 8x4096x40480:
11.940874 -> 9.134815 ms, 23.499612% reduction.
One warmed pair, complete attention (all heads and all queries):
2.579385 -> 2.481450 s, 3.796831% reduction.
Peak allocated 7.443359 GiB in both measured arms; the helper retains an
extra score buffer from its preceding softmax test. This is not production
attention memory or workflow peak, and there is no memory-saving claim.

Only N40480 dispatches to `_chunk_store_scale_softmax`. N5824/N10120 keep
`_cached_scale_softmax`; QK/PV remain rocBLAS. Unsupported shapes/features
keep their original guards/fallbacks. Prepared production function AST is
identical to the measured candidate after normalizing its function name.
Ten focused library tests cover the three lengths and dispatch selection.

The LTX INT8 constants candidate was rejected before GPU launches: corrected
compile private20 -> private40 bytes and hot-loop folded spill/reload 0 -> 2/4.
Its code is retained in research branch `mi50/ltx-store-int8-rejected`.
No INT8 production code changes are included in this accepted branch.
