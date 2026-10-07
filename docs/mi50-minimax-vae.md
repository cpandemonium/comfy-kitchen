# MI50 MiniMax video VAE experiments

`MI50_MINIMAX_VAE_INT8_TILES=1` selects measured integer GEMM tiles only for
contiguous FP16 inference on gfx906, INT8 weights, FP16 output and these exact
post-activation M/N/K forms:

- M = 3594 or 7188, N/K = 16384/2048, 6144/2048, 2048/8192 or 2048/2048.
- Default tiles: 128 x 256 x 32, group 8, 8 warps, 2 stages.
- M=3594 with N=2048: 128 x 128 x 64, group 8, 4 warps, 2 stages.

Other shapes, dtypes, devices, grad mode and disabled flags retain upstream
autotuning. Existing LTX and FP32 MiniMax transformer flags are independent.
ConvRot, activation, row quantization and output epilogues are unchanged.

On saved finite MiniMax AV latents (1152 x 640, 243 frames), one warmed process
ran A1=209.225 s, B1=202.452 s, B2=202.443 s, A2=209.312 s after a separate
warmup. The mean complete video decode improved 3.259% (6.821 s). Each B run
selected 12096 calls; all decoded frame tensors were bitwise identical to the
baseline. This measures video decoding, not the complete sampling workflow.
Native captured FP16 calls reproduced all eight forms exactly. Runtime matmul
precision flags were preserved, including actual FP16 accumulation.

`backends/eager/mi50_vae_attention.py` is an isolated research prototype. It is
not registered as a backend or connected to the application's VAE. Buffer
reuse, head groups 64/32/16/8/4/1 and contiguous K preserved outputs on the two
real Q/K/V captures but provided no meaningful speedup. The existing ordered
Wave64 softmax used directly at S=1797 changed some values and was rejected.
Do not infer whole-video equivalence or performance from attention replay.

Tests cover INT8 activation/ConvRot/bias/residual epilogues, independent flags
and safe fallbacks. The prototype has separate scope and exact buffer-reuse
tests. Deployment and AV file validation belong to the application's report.
