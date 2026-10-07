"""Experimental exact PyTorch buffer reuse for measured MI50 MiniMax VAE shapes."""

import os

import torch


def can_use(q, k, v, heads, mask=None, attn_precision=None, **kwargs):
    return (
        os.environ.get("MI50_MINIMAX_VAE_ATTENTION", "0") == "1"
        and not torch.is_grad_enabled()
        and torch.version.hip is not None
        and q.device.type == "cuda"
        and q.device == k.device == v.device
        and q.dtype == k.dtype == v.dtype == torch.float16
        and q.ndim == k.ndim == v.ndim == 4
        and q.shape == k.shape == v.shape
        and tuple(q.shape[1:]) == (32, 1797, 64)
        and q.shape[0] in (2, 4)
        and heads == 32
        and mask is None
        and attn_precision != torch.float32
        and kwargs.get("skip_reshape", False)
        and not kwargs.get("enable_gqa", False)
        and not kwargs.get("is_causal", False)
        and kwargs.get("dropout_p", 0.0) == 0.0
        and torch.cuda.get_device_properties(q.device).gcnArchName.startswith("gfx906")
    )


def attention(q, k, v, heads, mask=None, attn_precision=None, **kwargs):
    """Caller supplies the resolved attention precision and checks memory budget."""
    if not can_use(q, k, v, heads, mask, attn_precision, **kwargs):
        raise ValueError("requires measured gfx906 FP16 VAE inference shape")
    batch, _, tokens, dim = q.shape
    q, k, v = (t.reshape(batch * heads, tokens, dim) for t in (q, k, v))
    kt = k.transpose(-2, -1)
    if kwargs.get("contiguous_k", False):
        kt = kt.contiguous()
    head_chunk = kwargs.get("head_chunk", batch * heads)
    result = torch.empty_like(q)
    for start in range(0, batch * heads, head_chunk):
        end = start + head_chunk
        scores = torch.bmm(q[start:end], kt[start:end])
        scores.mul_(kwargs.get("scale", dim**-0.5))
        torch.softmax(scores, dim=-1, out=scores)
        torch.bmm(scores, v[start:end], out=result[start:end])
    if kwargs.get("skip_output_reshape", False):
        return result.reshape(batch, heads, tokens, dim)
    return (
        result.reshape(batch, heads, tokens, dim)
        .permute(0, 2, 1, 3)
        .reshape(batch, tokens, heads * dim)
    )
