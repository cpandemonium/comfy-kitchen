"""Keep chunking on the tested fc2 scope; verify native rounding and row tails."""

import pytest
import torch

pytest.importorskip("triton")
from comfy_kitchen.backends.triton.mi50_int8 import use_chunked_convrot
from comfy_kitchen.backends.triton.quantization import (
    _quantize_convrot_chunked,
    triton_quantize_rowwise,
)
from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation


def gfx906():
    return (
        torch.version.hip
        and torch.cuda.is_available()
        and torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx906")
    )


def test_cpu_scope(monkeypatch):
    monkeypatch.setenv("MI50_MINIMAX_CONVROT_CHUNKED", "1")
    monkeypatch.setenv("MI50_MINIMAX_INT8_TILES", "1")
    assert not use_chunked_convrot(
        torch.empty(1, 16), torch.empty(16, 16, dtype=torch.int8), torch.float32
    )


@pytest.mark.skipif(not gfx906(), reason="requires gfx906")
def test_gpu_scope_and_flags(monkeypatch):
    monkeypatch.setenv("MI50_MINIMAX_CONVROT_CHUNKED", "1")
    monkeypatch.setenv("MI50_MINIMAX_INT8_TILES", "1")
    x = torch.empty(55041, 14336, device="cuda", dtype=torch.float32)
    w = torch.empty(5376, 14336, device="cuda", dtype=torch.int8)
    assert not use_chunked_convrot(x[:52708], w, torch.float32)
    with torch.inference_mode():
        for m in (52672, 52708, 54200, 55040):
            assert use_chunked_convrot(x[:m], w, torch.float32)
        assert not use_chunked_convrot(x[:52671], w, torch.float32)
        assert not use_chunked_convrot(x, w, torch.float32)
        assert not use_chunked_convrot(x[:52708], w, torch.float16)
        assert not use_chunked_convrot(x[:52708], w, torch.float32, 64)
        with torch.autocast("cuda", dtype=torch.float16):
            assert not use_chunked_convrot(x[:52708], w, torch.float32)
        assert not use_chunked_convrot(x[:52708], w[:5375], torch.float32)
        assert not use_chunked_convrot(x[:52708, ::2], w[:, ::2], torch.float32)
        monkeypatch.setenv("MI50_MINIMAX_CONVROT_CHUNKED", "0")
        assert not use_chunked_convrot(x[:52708], w, torch.float32)
        monkeypatch.setenv("MI50_MINIMAX_CONVROT_CHUNKED", "1")
        monkeypatch.setenv("MI50_MINIMAX_INT8_TILES", "0")
        assert not use_chunked_convrot(x[:52708], w, torch.float32)


@pytest.mark.skipif(not gfx906(), reason="requires gfx906")
@pytest.mark.parametrize("rows", [2049, 4103])
def test_exact_rotation_quantization_with_tail(rows):
    torch.manual_seed(810)
    with torch.inference_mode():
        x = torch.randn(rows, 14336, device="cuda", dtype=torch.float32) * 0.2
        h = _build_hadamard(256, device=x.device, dtype=x.dtype)
        expected = triton_quantize_rowwise(_rotate_activation(x, h, 256))
        actual = _quantize_convrot_chunked(x, h, 256)
        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
