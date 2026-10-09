"""One 2048/8192 real fc2 pair after exact tail checks; no parameter sweep."""
import gc
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path('/comfy/mnt/benchmarks/minimax-convrot-20261009')
OLD = Path('/comfy/mnt/benchmarks/minimax-w4a8-20261008')
sys.path[:0] = ['/opt/ComfyUI', '/opt/mi50']
import torch
from launch import configure_runtime
configure_runtime()
from comfy_kitchen.backends.triton import quantization as quant, w4a8_int8
from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation

def idle():
    q = json.load(urllib.request.urlopen('http://127.0.0.1:8283/queue', timeout=5))
    assert not q['queue_running'] and not q['queue_pending']

def gpu(v):
    if isinstance(v, torch.Tensor): return v.cuda()
    if isinstance(v, dict): return {k: gpu(x) for k, x in v.items()}
    if isinstance(v, tuple): return tuple(gpu(x) for x in v)
    if isinstance(v, list): return [gpu(x) for x in v]
    return v

report = {'tails': [], 'arms': [], 'complete': False, 'accepted': False,
          'candidate': '8192 rows, same native FP32 rotation, padded tail and full GEMM',
          'policy': 'Exact tails first; one warmup and one timed real fc2 per arm.'}
def write(): (ROOT/'chunk8192-probe.json').write_text(json.dumps(report, indent=2)+'\n')

idle()
os.environ['MI50_MINIMAX_CONVROT_CHUNKED'] = '1'
native_chunk = quant._quantize_convrot_chunked
try:
    with torch.inference_mode():
        torch.manual_seed(810)
        for rows in (2049, 4103, 8193, 16391):
            idle()
            x = torch.randn(rows, 14336, device='cuda', dtype=torch.float32) * .2
            h = _build_hadamard(256, device=x.device, dtype=x.dtype)
            expected = quant.triton_quantize_rowwise(_rotate_activation(x, h, 256))
            actual = native_chunk(x, h, 256, chunk_rows=8192)
            row = {'rows': rows, 'quant_exact': torch.equal(expected[0], actual[0]),
                   'scales_exact': torch.equal(expected[1], actual[1])}
            report['tails'].append(row); write(); print('TAIL', json.dumps(row), flush=True)
            if not row['quant_exact'] or not row['scales_exact']:
                report.update(complete=True, rejection='Numerical tail gate failed')
                write(); sys.exit(0)
            del x, expected, actual
            gc.collect(); torch.cuda.empty_cache()
        data = gpu(torch.load(OLD/'w4a8-52708x5376x14336.pt', map_location='cpu', weights_only=False))
        reference = reference_q = reference_s = None
        for rows in (2048, 8192):
            stash = []
            def chunk(x, h, group_size, _rows=rows):
                value = native_chunk(x, h, group_size, chunk_rows=_rows)
                stash[:] = value
                return value
            quant._quantize_convrot_chunked = chunk
            def fn(): return w4a8_int8.w4a8_int8_linear(*data['args'], **data['kwargs'])
            idle(); out = fn(); torch.cuda.synchronize(); del out
            stash.clear(); gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            idle(); a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            started = time.perf_counter(); a.record(); host_started = time.perf_counter()
            out = fn(); host_s = time.perf_counter()-host_started
            b.record(); b.synchronize()
            row = {'rows': rows, 'gpu_s': a.elapsed_time(b)/1000,
                   'host_call_s': host_s, 'wall_s': time.perf_counter()-started,
                   'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
                   'peak_reserved_gib': torch.cuda.max_memory_reserved()/2**30,
                   'finite': bool(torch.isfinite(out).all())}
            if reference is None:
                reference, reference_q, reference_s = out.cpu(), stash[0].cpu(), stash[1].cpu()
            else:
                row.update(output_exact=torch.equal(reference, out.cpu()),
                           quant_exact=torch.equal(reference_q, stash[0].cpu()),
                           scales_exact=torch.equal(reference_s, stash[1].cpu()))
            del out; stash.clear()
            report['arms'].append(row); write(); print('FC2', json.dumps(row), flush=True)
        a, b = report['arms']
        report['gain_percent'] = 100*(1-b['gpu_s']/a['gpu_s'])
        report['accepted'] = (all(b.get(k) for k in ('finite', 'output_exact', 'quant_exact', 'scales_exact'))
                              and report['gain_percent'] > 1
                              and b['peak_allocated_gib'] <= a['peak_allocated_gib'] + .05)
        report['complete'] = True; write(); print('RESULT', json.dumps(report), flush=True)
finally:
    quant._quantize_convrot_chunked = native_chunk
