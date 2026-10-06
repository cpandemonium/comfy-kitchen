"""Regression coverage for gfx906 eager NA and bounded math-SDPA groups."""
from contextlib import contextmanager
import importlib

import pytest
import torch

na = importlib.import_module("comfy_kitchen.backends.eager.na")


def _gfx906_available():
    return (
        torch.version.hip is not None
        and torch.cuda.is_available()
        and torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx906")
    )


@contextmanager
def _math_only():
    backends = torch.backends.cuda
    states = [(backends.enable_flash_sdp, backends.flash_sdp_enabled()),
              (backends.enable_mem_efficient_sdp, backends.mem_efficient_sdp_enabled()),
              (backends.enable_cudnn_sdp, backends.cudnn_sdp_enabled()),
              (backends.enable_math_sdp, backends.math_sdp_enabled()),
              (backends.allow_fp16_bf16_reduction_math_sdp,
               backends.fp16_bf16_reduction_math_sdp_allowed())]
    try:
        backends.enable_flash_sdp(False)
        backends.enable_mem_efficient_sdp(False)
        backends.enable_cudnn_sdp(False)
        backends.enable_math_sdp(True)
        backends.allow_fp16_bf16_reduction_math_sdp(True)
        yield
    finally:
        for setter, value in states:
            setter(value)


@pytest.mark.skipif(not _gfx906_available(), reason="requires a real HIP gfx906 GPU")
@pytest.mark.parametrize("shape,kernel,causal,scale", [
    ((1, 5, 7, 9, 4, 8), [3, 3, 3], [False, False, False], None),
    ((2, 4, 5, 7, 2, 16), [3, 5, 3], [True, False, True], None),
    ((1, 2, 3, 4, 2, 8), [11, 11, 11], [False, False, False], 0.),
])
def test_gfx906_matches_sdpa_with_strided_inputs(monkeypatch, shape, kernel, causal, scale):
    torch.manual_seed(42)
    # Slice W to exercise a genuine non-contiguous input layout.
    expanded = (*shape[:3], shape[3] * 2, *shape[4:])
    tensors = [torch.randn(expanded, device="cuda", dtype=torch.float16)[:, :, :, ::2]
               for _ in range(3)]
    monkeypatch.setattr(na, "NA_SCORE_BUDGET", 4096)
    with _math_only(), torch.inference_mode():
        monkeypatch.setenv("MI50_NA_FAST_MATH", "0")
        reference = na.na3d(*tensors, kernel_size=kernel, is_causal=causal, scale=scale)
        monkeypatch.setenv("MI50_NA_FAST_MATH", "1")
        assert na._use_mi50_fast_math(*tensors)
        candidate = na.na3d(*tensors, kernel_size=kernel, is_causal=causal, scale=scale)
    assert torch.isfinite(candidate).all()
    torch.testing.assert_close(candidate, reference, rtol=0, atol=0)


def test_math_groups_do_not_stack_query_tiles(monkeypatch):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensors = [torch.randn(2, 4, 8, 8, 2, 8, device=device) for _ in range(3)]
    monkeypatch.setenv("MI50_NA_FAST_MATH", "0")
    monkeypatch.setattr(na, "NA_SCORE_BUDGET", 512)
    standard = torch.nn.functional.scaled_dot_product_attention
    batch_sizes = []

    def recording_sdpa(q, k, v, **kwargs):
        batch_sizes.append(q.shape[0])
        return standard(q, k, v, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", recording_sdpa)
    with _math_only(), torch.inference_mode():
        output = na.na3d(*tensors, kernel_size=[3, 3, 3])
    assert torch.isfinite(output).all()
    assert len(batch_sizes) > 1
    assert max(batch_sizes) == 2  # Input batch B, not multiple tile groups times B.


@pytest.mark.skipif(not _gfx906_available(), reason="requires a real HIP gfx906 GPU")
def test_fast_math_groups_stack_when_memory_allows(monkeypatch):
    tensors = [torch.randn(1, 8, 16, 16, 4, 8, device="cuda", dtype=torch.float16)
               for _ in range(3)]
    monkeypatch.setenv("MI50_NA_FAST_MATH", "1")
    monkeypatch.setattr(na, "NA_SCORE_BUDGET", 4096)
    monkeypatch.setattr(na, "_cuda_free_bytes", lambda device: 32 * 1024 ** 3)
    batches = []
    original = na._finite_mask_attention

    def recording_fast(q, k, v, mask):
        batches.append(q.shape[0])
        return original(q, k, v, mask)

    monkeypatch.setattr(na, "_finite_mask_attention", recording_fast)
    with _math_only(), torch.inference_mode():
        assert na._use_mi50_fast_math(*tensors)
        output = na.na3d(*tensors, kernel_size=[3, 3, 3])
    assert torch.isfinite(output).all()
    assert batches
    assert max(batches) > tensors[0].shape[0]


@pytest.mark.skipif(not _gfx906_available(), reason="requires a real HIP gfx906 GPU")
def test_fast_math_grouped_matches_ungrouped(monkeypatch):
    torch.manual_seed(42)
    tensors = [torch.randn(1, 8, 16, 16, 4, 8, device="cuda", dtype=torch.float16)
               for _ in range(3)]
    monkeypatch.setenv("MI50_NA_FAST_MATH", "1")
    monkeypatch.setattr(na, "NA_SCORE_BUDGET", 4096)
    with _math_only(), torch.inference_mode():
        monkeypatch.setattr(na, "_cuda_free_bytes", lambda device: 0)
        ungrouped = na.na3d(*tensors, kernel_size=[3, 3, 3])
        monkeypatch.setattr(na, "_cuda_free_bytes", lambda device: 32 * 1024 ** 3)
        grouped = na.na3d(*tensors, kernel_size=[3, 3, 3])
    assert torch.isfinite(grouped).all()
    torch.testing.assert_close(grouped, ungrouped, rtol=1e-3, atol=1e-3)


def test_training_fp32_preserves_outputs_and_gradients(monkeypatch):
    torch.manual_seed(42)
    values = [torch.randn(1, 2, 3, 4, 2, 8) for _ in range(3)]
    results = []
    for enabled in ("0", "1"):
        monkeypatch.setenv("MI50_NA_FAST_MATH", enabled)
        tensors = [t.clone().requires_grad_() for t in values]
        assert not na._use_mi50_fast_math(*tensors)
        output = na.na3d(*tensors, kernel_size=[3, 3, 3], is_causal=[True, False, False])
        output.square().sum().backward()
        gradients = [t.grad.clone() for t in tensors]
        assert all(torch.isfinite(g).all() for g in gradients)
        results.append((output.detach(), gradients))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    for reference, candidate in zip(results[0][1], results[1][1], strict=True):
        torch.testing.assert_close(reference, candidate, rtol=0, atol=0)
