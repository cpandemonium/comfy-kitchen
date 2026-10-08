"""Measured exact INT8 tiles for MI50 LTX, MiniMax transformer and video VAE."""

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
_VAE_SHAPES = {
    (m, n, k)
    for m in (3594, 7188)
    for n, k in ((16384, 2048), (2048, 8192), (6144, 2048), (2048, 2048))
}
_VAE_SMALL = {(3594, 2048, 8192), (3594, 2048, 2048)}
_VAE_SMALL_CONFIG = dict(_CONFIG, block_n=128, block_k=64, num_warps=4)


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
    shape = (x.shape[0], weight.shape[0], x.shape[1])
    vae = (
        os.environ.get("MI50_MINIMAX_VAE_INT8_TILES", "0") == "1"
        and x.dtype == torch.float16
        and out_dtype == torch.float16
        and shape in _VAE_SHAPES
    )
    if not (ltx or minimax or vae):
        return None
    if not torch.cuda.get_device_properties(x.device).gcnArchName.startswith("gfx906"):
        return None
    return dict(_VAE_SMALL_CONFIG if vae and shape in _VAE_SMALL else _CONFIG)


def use_chunked_convrot(x, weight, out_dtype, group_size=256):
    """Limit the memory change to the measured FP32 MiniMax fc2 projection."""
    return (
        os.environ.get("MI50_MINIMAX_CONVROT_CHUNKED", "0") == "1"
        and group_size == 256
        and not torch.is_autocast_enabled("cuda")
        and tuple(weight.shape) == (5376, 14336)
        and select_config(x, weight, out_dtype) is not None
        and x.dtype == torch.float32
        and out_dtype == torch.float32
    )


def use_nolicm_vae_gemm(x, weight, out_dtype):
    """Test only the two dominant measured video VAE fc1 shapes."""
    return (
        os.environ.get("MI50_MINIMAX_VAE_GEMM_NO_LICM", "0") == "1"
        and x.shape[0] in (3594, 7188)
        and tuple(weight.shape) == (16384, 2048)
        and x.dtype == torch.float16
        and out_dtype == torch.float16
        and not torch.is_autocast_enabled("cuda")
        and select_config(x, weight, out_dtype) is not None
    )
