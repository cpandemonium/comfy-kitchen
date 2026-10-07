"""Scope and exact numerical checks for measured gfx906 kernel variants."""

import pytest
import torch

pytest.importorskip("triton")
from comfy_kitchen.backends.triton.mi50_int8 import select_config
from comfy_kitchen.backends.triton.mi50_softmax import scale_softmax_inplace
from comfy_kitchen.backends.triton.quantization import int8_linear


def _gfx906():
    return (
        torch.version.hip
        and torch.cuda.is_available()
        and torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx906")
    )


def test_cpu_keeps_original_kernels(monkeypatch):
    monkeypatch.setenv("MI50_LTX_INT8_TILES", "1")
    assert (
        select_config(torch.empty(1, 16), torch.empty(16, 16, dtype=torch.int8), torch.float16)
        is None
    )
    with pytest.raises(ValueError, match="gfx906"):
        scale_softmax_inplace(torch.ones(2, 40480, dtype=torch.float16), 128**-0.5)


@pytest.mark.skipif(not _gfx906(), reason="requires gfx906")
def test_int8_scope(monkeypatch):
    monkeypatch.setenv("MI50_LTX_INT8_TILES", "1")
    x = torch.empty(40480, 4096, device="cuda", dtype=torch.float16)
    weight = torch.empty(4096, 4096, device="cuda", dtype=torch.int8)
    assert select_config(x, weight, torch.float16) is None  # Training/grad mode.
    with torch.inference_mode():
        assert select_config(x, weight, torch.float16)["block_k"] == 32
        assert select_config(x[:10120], weight, torch.float16) is None
        assert select_config(x, weight, torch.float32) is None
        assert select_config(x[:, ::2], weight[:, :2048], torch.float16) is None
        monkeypatch.setenv("MI50_LTX_INT8_TILES", "0")
        assert select_config(x, weight, torch.float16) is None


@pytest.mark.skipif(not _gfx906(), reason="requires gfx906")
@pytest.mark.parametrize("per_channel", [False, True])
def test_exact_int8_epilogue_and_residual(per_channel, monkeypatch):
    torch.manual_seed(37)
    with torch.inference_mode():
        x = torch.randn(40480, 4096, device="cuda", dtype=torch.float16) * 0.15
        weight = torch.randint(-127, 128, (4096, 4096), device="cuda", dtype=torch.int8)
        scale = torch.full((4096,) if per_channel else (1,), 0.0002, device="cuda")
        bias = torch.randn(4096, device="cuda", dtype=torch.float16) * 0.05
        residual = x * 0.1
        residual_scale = torch.full((4096,), 0.8, device="cuda", dtype=torch.float16)
        kwargs = {
            "bias": bias,
            "out_dtype": torch.float16,
            "convrot": True,
            "residual": residual,
            "residual_scale": residual_scale,
        }
        monkeypatch.setenv("MI50_LTX_INT8_TILES", "0")
        reference = int8_linear(x, weight, scale, **kwargs)
        monkeypatch.setenv("MI50_LTX_INT8_TILES", "1")
        result = int8_linear(x, weight, scale, **kwargs)
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


@pytest.mark.skipif(not _gfx906(), reason="requires gfx906")
def test_ordered_softmax_exact_and_inplace():
    torch.manual_seed(56)
    with torch.inference_mode():
        scores = torch.randn(128, 40480, device="cuda", dtype=torch.float16) * 3
        scores[0].zero_()
        scores[1].fill_(-600)
        scores[1, 20000] = 600
        reference = (scores * (128**-0.5)).softmax(-1)
        result = scale_softmax_inplace(scores, 128**-0.5)
        assert result.data_ptr() == scores.data_ptr()
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
        with pytest.raises(ValueError, match="contiguous"):
            scale_softmax_inplace(scores[:, :40000], 128**-0.5)
