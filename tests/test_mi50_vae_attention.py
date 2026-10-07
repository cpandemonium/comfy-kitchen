"""Numerical and fallback checks for the independent MiniMax VAE buffer path."""

import pytest
import torch

from comfy_kitchen.backends.eager.mi50_vae_attention import attention, can_use


def _gfx906():
    return (
        torch.version.hip
        and torch.cuda.is_available()
        and torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx906")
    )


def test_cpu_rejected(monkeypatch):
    monkeypatch.setenv("MI50_MINIMAX_VAE_ATTENTION", "1")
    q = torch.empty(2, 32, 1797, 64, dtype=torch.float16)
    assert not can_use(q, q, q, 32, skip_reshape=True)


@pytest.mark.skipif(not _gfx906(), reason="requires gfx906")
def test_attention_scope(monkeypatch):
    monkeypatch.setenv("MI50_MINIMAX_VAE_ATTENTION", "1")
    q = torch.empty(2, 32, 1797, 64, device="cuda", dtype=torch.float16)
    assert not can_use(q, q, q, 32, skip_reshape=True)
    with torch.inference_mode():
        assert can_use(q, q, q, 32, skip_reshape=True)
        assert not can_use(q, q, q, 32)
        assert not can_use(q, q, q, 32, skip_reshape=True, attn_precision=torch.float32)
        assert not can_use(q, q, q, 32, skip_reshape=True, mask=torch.ones(1797, device="cuda"))
        assert not can_use(q, q, q, 32, skip_reshape=True, enable_gqa=True)
        assert not can_use(q, q, q, 32, skip_reshape=True, is_causal=True)
        assert not can_use(q, q, q, 32, skip_reshape=True, dropout_p=0.1)
        assert not can_use(q[:1], q[:1], q[:1], 32, skip_reshape=True)
        monkeypatch.setenv("MI50_MINIMAX_VAE_ATTENTION", "0")
        assert not can_use(q, q, q, 32, skip_reshape=True)


@pytest.mark.skipif(not _gfx906(), reason="requires gfx906")
@pytest.mark.parametrize("batch", [2, 4])
def test_exact_softmax_buffer_reuse(batch, monkeypatch):
    monkeypatch.setenv("MI50_MINIMAX_VAE_ATTENTION", "1")
    torch.manual_seed(93)
    with torch.inference_mode():
        # Match the real VAE's transposed, non-contiguous Q/K/V layout.
        q, k, v = (
            torch.randn(batch, 1797, 32, 64, device="cuda", dtype=torch.float16).transpose(1, 2)
            for _ in range(3)
        )
        q[:, :, 0].zero_()
        qp, kp, vp = (t.reshape(batch * 32, 1797, 64) for t in (q, k, v))
        scores = torch.einsum("bid,bjd->bij", qp, kp) * 0.125
        reference = torch.einsum("bij,bjd->bid", scores.softmax(-1), vp)
        expected = reference.reshape(batch, 32, 1797, 64).transpose(1, 2).reshape(batch, 1797, 2048)
        result = attention(q, k, v, 32, skip_reshape=True)
        assert bool(torch.isfinite(result).all())
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            attention(q, k, v, 32, skip_reshape=True, skip_output_reshape=True),
            reference.reshape(batch, 32, 1797, 64),
            rtol=0,
            atol=0,
        )
