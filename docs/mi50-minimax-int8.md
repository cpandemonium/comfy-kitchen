# Exact MiniMax INT8 tiles on gfx906

`MI50_MINIMAX_INT8_TILES=1` opts inference into measured tiles for FP32
MiniMax-H3 activations and INT8 ConvRot weights on AMD gfx906. It is independent
of `MI50_LTX_INT8_TILES`; both are disabled by default in the library.

The guard requires contiguous two-dimensional CUDA input and weights, the same
device, FP32 input/output, disabled gradients, 52672–55040 input rows, and one
of the following weight `(N, K)` shapes:

* `(21504, 5376)`
* `(28672, 5376)`
* `(5376, 14336)`
* `(5376, 7168)`

These are joint video/audio/text sequences from a 1152×640, 243-frame workflow.
Prompt length changes M. The FFN-down raw input width is 28672, but SwiGLU
halves it before selection, yielding GEMM K=14336. Other dimensions, dtypes,
devices and training retain upstream autotuning.

The selected tile is 128×256×32, group size 8, eight warps, two stages.
It uses the existing integer accumulation and epilogue, preserving ConvRot,
bias, per-channel/per-tensor scales and residual behavior. No attention
algorithm, sparse block selection or model precision is changed.

On finite real layer activations at M=52708, native linear call medians decreased
by 5.8–21.6%, with bitwise-identical outputs. Neighbor lengths 52672, 53248 and
55040 decreased by 5.4–7.3%, also bitwise identical. Two noisy three-repeat
measurements were repeated seven times after two warmups: 6.3% and 7.1%.
These percentages concern individual linear calls and are not whole-workflow
speedups. A full four-step run preserved both video and audio latents bitwise.
The profiled GEMM ranges decreased from 111.388 to 106.589 seconds, but SLA
ranges increased and the clean fourth step took 381.881 versus 372.854 seconds.
The production deployment therefore keeps this option disabled: a warmed
whole-sampler speedup has not been established. A lower first-step time also
includes differences in autotuning/cache state and is not a warmed speedup.

Run `tests/test_mi50_kernel_tuning.py` on gfx906 to check opt-in scope,
independence from LTX, and exact bias/ConvRot/residual epilogues with both scale
layouts. Frozen real replay and full-model comparison complement these checks.
