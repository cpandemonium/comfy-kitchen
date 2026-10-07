"""Measured tiles for FP16 ConvRot INT8 inference on the MI50 LTX refiner."""

import os
import torch

_SHAPES = {
    (40480, 16384, 4096),
    (40480, 2048, 4096),
    (40480, 4096, 16384),
    (40480, 4096, 2048),
    (40480, 4096, 4096),
}
_CONFIG = {
    "block_m": 128,
    "block_n": 256,
    "block_k": 32,
    "group_size_m": 8,
    "num_warps": 8,
    "num_stages": 2,
}


def select_config(x, weight, out_dtype):
    """Return a verified tile; other devices/shapes retain upstream autotuning."""
    if (
        os.environ.get("MI50_LTX_INT8_TILES", "0") != "1"
        or torch.is_grad_enabled()
        or not torch.version.hip
        or x.device.type != "cuda"
        or x.dtype != torch.float16
        or out_dtype != torch.float16
        or weight.dtype != torch.int8
        or weight.device != x.device
        or x.ndim != 2
        or weight.ndim != 2
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or (x.shape[0], weight.shape[0], x.shape[1]) not in _SHAPES
        or x.shape[1] != weight.shape[1]
    ):
        return None
    if not torch.cuda.get_device_properties(x.device).gcnArchName.startswith("gfx906"):
        return None
    return dict(_CONFIG)
