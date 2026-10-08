# Scoped MiniMax fc2 ConvRot memory reduction

The FP32 MiniMax W4A8 fc2 projection can rotate and row-quantize 2048 activation
rows at a time instead of allocating one full rotated matrix. Native FP32
Hadamard matmul, INT8 row quantization, scales, complete GEMM and epilogue stay
the same. The last partial rotation is zero-padded to 2048 rows: an unpadded
seven-row tail selected different native rounding and changed two quantized
elements in a random numerical check. Padding preserves the tested rounding.

Opt in with `MI50_MINIMAX_CONVROT_CHUNKED=1`. The guard also requires the existing
`MI50_MINIMAX_INT8_TILES=1` scope: gfx906, inference without autocast, contiguous
FP32 input/output, contiguous same-device INT8 weight `[5376,14336]`,
M in `[52672,55040]`, and ConvRot group size 256. All other paths retain the
original rotation. LTX and FP16 MiniMax VAE are outside the guard.

On real saved W4A8 fc2 `[52708,14336]`, one warmup and one timed complete linear
per arm measured 0.769773 -> 0.666188 seconds (-13.457%). Process peak allocated
was 7.502243 -> 4.687332 GiB, saving 2.814911 GiB. Activation INT8 data, row
scales and full output matched bitwise and were finite. These are isolated-layer
measurements, not whole-workflow speed or memory guarantees. A full long-prompt
sampler remains a user workload check.

`tests/test_mi50_chunked_convrot.py` covers guard boundaries, disabled flags,
autocast, group size, incompatible output/weight/layout and one/seven-row tails.
Existing nine Triton JIT function ASTs were compared with the prior pinned
version and remain identical.

Research also tested VAE INT8 GEMM no-LICM on `[7188,16384,2048]` and
`[3594,16384,2048]`: exact outputs but 29-31% slower, so its production source
was removed. Reproduce that candidate from research commit
`347f11b` on branch `mi50/minimax-memory-query32-research`; its sample consumes
server-side captures. The source removal is intentional.
