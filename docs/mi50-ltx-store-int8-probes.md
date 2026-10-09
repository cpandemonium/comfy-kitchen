# Rejected LTX chunked softmax stores and INT8 dimension specialization

Compile-only on gfx906, 2026-10-09, Triton 3.8.0, pinned production 4d651589.
These samples are research kernels and are not registered or installed.
No GPU numerical or timing calls were made for either library candidate.

`samples/mi50_ltx_chunk_store_probe.py` retains the ordered FP16 score cache
but calculates/stores eight probabilities at a time. N40480 compilation:
production VGPR125/private0/spill0, candidate VGPR128/private20/spill4.
The candidate adds scratch despite reducing total assembly instructions
2120 to 1421. The static gate rejected it; lower static instruction count
does not establish better runtime performance.

`samples/mi50_ltx_int8_constants_probe.py` fixes N16384/K4096 and contiguous
strides, retains INT32 dot/FP32 dequantization and the existing 128x256x32
tile, removes wrapped row indices, bounds the M tail, and removes K masks.
Compilation models the real M40480 divisibility and buffers below 2 GiB.
Production private20/spill4 becomes private40/spill9; hot-loop instructions
608 become 609, with folded spill/reload operations 0 becoming 2/4.
Both use 32768 bytes LDS. The static gate rejected it.

An initial generic compile omitted the actual M divisibility and output
buffer range; it was corrected before deciding. The rejected conclusion
uses the corrected compilation, not that generic comparison.

Production kernels, guards, flags and source pin remain unchanged. A
separate project experiment batching two VideoVAE Conv3D spatial tiles
also failed its timing gate after exact output validation. Measurements
and project harness corrections are recorded in the parent project.
