# gfx906 eager neighborhood attention

This branch is based on upstream `v0.2.37` (`be003b7c23c5b01328657955b8bc5d3f073d868e`).
It contains the two changes validated with LTX 2.5 VideoVAE on one AMD MI50 32 GiB:

- Bound math-SDPA query-tile groups to one tile. HIP uses PyTorch's `cuda`
  device type, but math SDPA materializes scores and cannot use the original
  grouping budget designed for fused CUDA attention.
- Use explicit QK, additive mask, softmax and AV for FP16 inference on gfx906.
  The eager NA implementation has finite additive masks and a valid window
  for every query. With finite logits it does not need general SDPA's checks
  for rows containing only negative infinity.

The fast path applies only to HIP gfx906, FP16, disabled gradients, math-only
SDPA and enabled FP16/BF16 math reduction. It does not change global PyTorch
SDPA, other architectures, FP32/BF16 or training. Set `MI50_NA_FAST_MATH=0`
to select standard SDPA while retaining the bounded math grouping.

The finite-logit requirement matters: this is not a general attention helper
for arbitrary masks, infinities or NaNs. The validated model uses finite
inputs and the mask produced by `_group_mask`.

## Validation

The original deployment used PyTorch `2.15.0a0+rocm10.2.0a20261005`, HIP
`7.17.26392`, FP16 math reduction and FP16 accumulation. All eight LTX decode
tile sizes were compared with the standard SDPA path:

| Latent T x H x W | SDPA seconds | Fast seconds | Speedup |
| --- | ---: | ---: | ---: |
| 8 x 16 x 16 | 25.67 | 20.50 | 1.252x |
| 8 x 16 x 12 | 18.76 | 15.39 | 1.219x |
| 8 x 8 x 16 | 12.84 | 10.35 | 1.241x |
| 8 x 8 x 12 | 9.23 | 7.62 | 1.211x |
| 4 x 16 x 16 | 11.75 | 9.98 | 1.178x |
| 4 x 16 x 12 | 9.05 | 7.36 | 1.230x |
| 4 x 8 x 16 | 5.81 | 4.93 | 1.179x |
| 4 x 8 x 12 | 4.64 | 3.86 | 1.201x |

Every tile output was finite and bitwise equal. Full tiled decoding and MP4
writing for 1280 x 704, 361 frames at 24 fps improved from 799.628 to 654.382
seconds (18.16% less time). All decoded video frames had the same SHA-256.
Observed device VRAM peak was 9.44 GiB, sampled every three seconds. These
are workload measurements, not guarantees for other models or hardware.

`tests/test_na_mi50.py` covers strided inputs, causal windows, edge windows,
batch > 1, oversized kernels, zero scale, bounded grouping, and unchanged
FP32 training outputs and gradients. GPU parity tests skip on other devices.

```bash
python -m pytest tests/test_na_mi50.py
```

For an MI50 deployment, retain the version-matched upstream wheel's native
modules and install this branch's `comfy_kitchen/backends/eager/na.py` from a
pinned Git commit. This avoids rebuilding unrelated CUDA/HIP extensions on
a runtime-only ROCm image.
