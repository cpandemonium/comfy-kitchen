"""Scope and numerical regression checks for the gfx906 long-video kernel."""
import pytest
import torch
from comfy_kitchen.backends.eager import mi50_attention as attention


def _gfx906():
    return (torch.version.hip is not None and torch.cuda.is_available()
            and torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx906"))


def test_cpu_and_training_keep_original_backend():
    q = torch.randn(1, 16, 128, requires_grad=True)
    assert not attention.can_use(q, q, q, 1)
    with pytest.raises(ValueError, match="validated gfx906"):
        attention.long_attention(q, q, q, 1)


@pytest.mark.skipif(not _gfx906(), reason="requires HIP gfx906")
def test_first_pass_and_audio_shapes_keep_original_backend():
    with torch.inference_mode():
        for n, c, heads in [(10120, 4096, 32), (376, 2048, 32), (1024, 4096, 32)]:
            q = torch.empty(1, n, c, device="cuda", dtype=torch.float16)
            assert not attention.can_use(q, q, q, heads)


@pytest.mark.skipif(not _gfx906(), reason="requires HIP gfx906")
def test_masks_precision_and_other_features_keep_original_backend():
    q = torch.empty(1, 40480, 4096, device="cuda", dtype=torch.float16)
    assert not attention.can_use(q, q, q, 32)  # Grad mode is enabled.
    with torch.inference_mode():
        assert attention.can_use(q, q, q, 32)
        for options in [dict(mask=torch.ones(1, device="cuda")),
                        dict(attn_precision=torch.float32), dict(skip_reshape=True),
                        dict(enable_gqa=True), dict(is_causal=True), dict(dropout_p=.1)]:
            assert not attention.can_use(q, q, q, 32, **options)
        assert not attention.can_use(q, q, q, 16)
        with pytest.raises(ValueError, match="even one attention row"):
            attention.long_attention(q, q, q, max_score_bytes=1)


@pytest.mark.skipif(not _gfx906(), reason="requires HIP gfx906")
def test_dense_full_refiner_attention_matches_original_fp16_math():
    torch.manual_seed(42)
    with torch.inference_mode():
        values = [torch.randn(1,40480,4096,dtype=torch.float16,device="cuda")*.4 for _ in range(3)]
        q,k,v = [x.reshape(1,40480,32,128).permute(0,2,1,3).reshape(32,40480,128).contiguous()
                 for x in values]
        result = torch.empty_like(q)
        scale = 128**-.5
        for start in range(0,40480,1265):
            scores = torch.bmm(q[:,start:start+1265],k.transpose(-2,-1))*scale
            probabilities = scores.softmax(-1)
            del scores
            result[:,start:start+1265] = torch.bmm(probabilities,v)
            del probabilities
        reference = result.reshape(1,32,40480,128).permute(0,2,1,3).reshape(1,40480,4096)
        del q,k,v,result
        torch.cuda.reset_peak_memory_stats()
        allocated_before = torch.cuda.memory_allocated()
        control = attention.long_attention(*values, 32, fused_softmax=False, inplace_softmax=False)
        control_extra = torch.cuda.max_memory_allocated() - allocated_before
        torch.testing.assert_close(control, reference, rtol=0, atol=0)
        del control
        torch.cuda.reset_peak_memory_stats()
        allocated_before = torch.cuda.memory_allocated()
        output = attention.long_attention(*values, 32, fused_softmax=False, inplace_softmax=True)
        candidate_extra = torch.cuda.max_memory_allocated() - allocated_before
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output, reference, rtol=0, atol=0)
        # Score/probability aliasing must remove a real temporary allocation,
        # not merely advertise a lower budget to the caller.
        assert control_extra - candidate_extra > 2 * 1024**3
        ordered = attention.long_attention(*values, 32, ordered_softmax=True)
        torch.testing.assert_close(ordered, output, rtol=0, atol=0)
        del ordered
        fused = attention.long_attention(*values, 32, fused_softmax=True)
        assert torch.isfinite(fused).all()
        torch.testing.assert_close(fused, output, rtol=1e-3, atol=1e-3)
