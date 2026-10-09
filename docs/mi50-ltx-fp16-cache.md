# LTX ordered softmax: FP16 register cache on gfx906

Measured on 2026-10-09, ROCm PyTorch 2.15.0a0 and Triton 3.8.0, MI50 32 GiB.
The kernel retains rounded FP16 scores in two explicitly distributed Gluon
tensors (64 + 16 slots per logical thread), computes exponentials in FP32,
and preserves the serial chunk/lane and descending Wave64 reduction order.
All keys participate. There is no extra global buffer or scratch allocation.
The old three-read kernel remains in the module for comparison and rollback.

Compilation at N40480: old VGPR56, cached VGPR125, both private0/spill0.
One warmed timing pair of 8x4096x40480 scores: 19.312139 -> 11.803752 ms
(38.879% reduction), with bitwise FP16 output equality.
One warmed complete attention pair (all 32 heads and all 40480 queries):
2.905996 -> 2.584268 seconds (11.071%), peak allocated 4.972656 GiB in both.
QK and PV retain rocBLAS; only scale/softmax changes.

N5824 additionally matches the native ComfyUI attention_split output bitwise:
84.057507 -> 83.137545 ms (1.094% in one pair). The main benefit at this
length is peak allocated 5.458496 -> 1.801758 GiB, including the same frozen
input tensors. This is attention memory, not the peak of a full workflow.
These small inputs are slices of a real N40480 capture, not a capture of the
latest smaller generation. Tests separately cover random scores, constant
rows, extreme finite scores, tails, inference/CPU/feature guards, and budget.
Only the two measured lengths are admitted; other lengths keep their backend.

The research PV candidate in samples/mi50_ltx_fp16_cache_pv_probe.py stages
FP16 P/V tiles in explicit LDS layouts. It compiles without scratch/spills,
but changes FP16 values relative to the configured rocBLAS accumulation:
32/21 differing values on long 32/17-query cases and 16/11 on small cases,
max absolute error 0.0009765625/0.00048828125. It is not registered or timed.
An exact production backend cannot adopt this accumulation order.

Measurements are isolated attention pairs, not end-to-end sampler speedups.
Full user generation is the subsequent integration check. Seven targeted
library tests passed before deployment; final installed checks are recorded
by the server configuration repository.
