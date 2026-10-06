# MI50 dense attention for the LTX 2.5 refiner

`comfy_kitchen.backends.eager.mi50_attention` implements the dense attention
shape used by a 1280 x 704, 361-frame LTX 2.5 second sampling pass:
Q/K/V `[1, 40480, 4096]`, 32 heads of width 128, FP16 inference on HIP gfx906.

The default path splits both heads and queries: eight heads and up to 4096
query tokens per group. Each group attends **every key**, using rocBLAS BMM,
in-place score scaling and ordinary PyTorch softmax. It changes neither the
attention algorithm nor the model, token count, sampling schedule or seed.
A caller-provided temporary-memory budget bounds the score-buffer size.
Output allocation and batched AV avoid unnecessary intermediate copies.

`can_use` restricts the fast path to the validated shape, architecture and
dtype. Training, CPU, FP32/BF16, masks, causal attention, GQA and other shapes
keep the caller's original implementation. Integration with ComfyUI should
apply the override only to LTX video self-attention and retain existing
attention overrides and the selected backend. In the validated workflow the
first pass has 10120 video tokens and does not use this path.

## Numerical validation

Real Q/K/V from transformer block 24 matched the original attention output
bitwise for the default tiling. A complete full-size refiner step also matched
both sampled video `[1,128,46,22,40]` and audio `[1,8,376,16]` latents bitwise.
The comparison used actual Q5_K_M weights, CFG 1, Euler ancestral and sigmas
`[0.85,0.725]`. The saved final video latent and zero audio represented the
original workload shape; they were not the unavailable original refiner input.

The tests cover the full dense-attention output and fallback scope:

```bash
python -m pytest tests/test_mi50_long_attention.py tests/test_na_mi50.py
```

The process-local `fused_softmax=True` research option uses a Wave64 Triton
scale/softmax kernel. It is **disabled by default and in deployment**: while
faster in isolation, its rounding differed and a full-model precision gate
failed. It also needs a host C compiler for Triton's launcher. The ordinary
softmax path needs no compiler and was selected for deployment.

This branch builds on the existing gfx906 neighborhood-attention fixes;
it does not alter their implementation or the VideoVAE decoder.
