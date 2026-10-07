"""Measured exact INT8 tiles for MI50 LTX and FP32 MiniMax inference."""

import os
import torch

_SHAPES = {
    (m, n, k)
    for m in (10120, 40480)
    for n, k in ((16384, 4096), (2048, 4096), (4096, 16384), (4096, 2048), (4096, 4096))
}
_CONFIG = {
    "block_m": 128,
    "block_n": 256,
    "block_k": 32,
    "group_size_m": 8,
    "num_warps": 8,
    "num_stages": 2,
}
_MINIMAX_NK = {(21504, 5376), (28672, 5376), (5376, 14336), (5376, 7168)}


def select_config(x, weight, out_dtype):
    """Return a verified tile; other devices/shapes retain upstream autotuning."""
    if (
        torch.is_grad_enabled()
        or not torch.version.hip
        or x.device.type != "cuda"
        or weight.dtype != torch.int8
        or weight.device != x.device
        or x.ndim != 2
        or weight.ndim != 2
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or x.shape[1] != weight.shape[1]
    ):
        return None
    ltx = (
        os.environ.get("MI50_LTX_INT8_TILES", "0") == "1"
        and x.dtype == torch.float16
        and out_dtype == torch.float16
        and (x.shape[0], weight.shape[0], x.shape[1]) in _SHAPES
    )
    minimax = (
        os.environ.get("MI50_MINIMAX_INT8_TILES", "0") == "1"
        and x.dtype == torch.float32
        and out_dtype == torch.float32
        and 52672 <= x.shape[0] <= 55040
        and tuple(weight.shape) in _MINIMAX_NK
    )
    if not (ltx or minimax):
        return None
    if not torch.cuda.get_device_properties(x.device).gcnArchName.startswith("gfx906"):
        return None
    return dict(_CONFIG)
