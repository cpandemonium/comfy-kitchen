"""One real W4A8 fc2 pair, including native activation quant/scales parity."""

import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path("/comfy/mnt/benchmarks/minimax-memory-query32-20261008")
OLD = Path("/comfy/mnt/benchmarks/minimax-w4a8-20261008")
sys.path[:0] = ["/opt/ComfyUI", "/opt/mi50"]
import torch  # noqa: E402
from launch import configure_runtime  # noqa: E402

configure_runtime()
from comfy_kitchen.backends.triton import quantization as base, mi50_int8, w4a8_int8  # noqa: E402


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


scope = module("memory_scope", ROOT / "mi50_int8.py")
candidate = module("memory_quantization", ROOT / "quantization.py")
mi50_int8.use_chunked_convrot = scope.use_chunked_convrot
os.environ["MI50_MINIMAX_CONVROT_CHUNKED"] = "1"
report = {
    "arms": [],
    "candidate": "2048 rows per native rotation; same row quant and full GEMM",
    "policy": "One warmup and one timed full W4A8 fc2 per arm.",
}


def idle():
    q = json.load(urllib.request.urlopen("http://127.0.0.1:8283/queue"))
    assert not q["queue_running"] and not q["queue_pending"]


def gpu(v):
    if isinstance(v, torch.Tensor):
        return v.cuda()
    if isinstance(v, dict):
        return {k: gpu(x) for k, x in v.items()}
    if isinstance(v, tuple):
        return tuple(gpu(x) for x in v)
    if isinstance(v, list):
        return [gpu(x) for x in v]
    return v


original = w4a8_int8.int8_linear
try:
    idle()
    urllib.request.urlopen(
        urllib.request.Request(
            "http://127.0.0.1:8283/free",
            data=b'{"unload_models":true,"free_memory":true}',
            headers={"Content-Type": "application/json"},
        ),
        timeout=10,
    ).close()
    with torch.inference_mode():
        data = gpu(
            torch.load(OLD / "w4a8-52708x5376x14336.pt", map_location="cpu", weights_only=False)
        )
        report["input_shape"] = list(
            data["args"][0].shape if data["args"] else data["kwargs"]["x"].shape
        )
        reference = None
        quant_reference = None
        scale_reference = None
        for label, mod, quant_name in [
            ("baseline", base, "triton_quantize_rowwise"),
            ("candidate", candidate, "_quantize_convrot_chunked"),
        ]:
            stash = []
            native = getattr(mod, quant_name)

            def quant(*args, _native=native, _stash=stash, **kwargs):
                value = _native(*args, **kwargs)
                _stash[:] = value
                return value

            setattr(mod, quant_name, quant)
            w4a8_int8.int8_linear = mod.int8_linear

            def fn():
                return w4a8_int8.w4a8_int8_linear(*data["args"], **data["kwargs"])

            idle()
            out = fn()
            torch.cuda.synchronize()
            del out
            stash.clear()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            idle()
            started = time.perf_counter()
            a.record()
            out = fn()
            b.record()
            b.synchronize()
            row = {
                "name": label,
                "gpu_s": a.elapsed_time(b) / 1000,
                "wall_s": time.perf_counter() - started,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "finite": bool(torch.isfinite(out).all()),
            }
            if reference is None:
                reference = out.cpu()
                quant_reference = stash[0].cpu()
                scale_reference = stash[1].cpu()
            else:
                row.update(
                    bitwise_equal=torch.equal(reference, out.cpu()),
                    quant_bitwise_equal=torch.equal(quant_reference, stash[0].cpu()),
                    scales_bitwise_equal=torch.equal(scale_reference, stash[1].cpu()),
                )
            setattr(mod, quant_name, native)
            del out
            stash.clear()
            report["arms"].append(row)
            (ROOT / "chunked-fc2-probe.json").write_text(json.dumps(report, indent=2) + "\n")
            print("FC2", json.dumps(row), flush=True)
        report["gain_percent"] = 100 * (1 - report["arms"][1]["gpu_s"] / report["arms"][0]["gpu_s"])
        report["allocated_saved_gib"] = (
            report["arms"][0]["peak_allocated_gib"] - report["arms"][1]["peak_allocated_gib"]
        )
        report["accepted"] = (
            all(
                report["arms"][1].get(k)
                for k in ("finite", "bitwise_equal", "quant_bitwise_equal", "scales_bitwise_equal")
            )
            and report["gain_percent"] > -1
        )
        report["complete"] = True
        (ROOT / "chunked-fc2-probe.json").write_text(json.dumps(report, indent=2) + "\n")
        print("RESULT", json.dumps({k: v for k, v in report.items() if k != "arms"}), flush=True)
finally:
    w4a8_int8.int8_linear = original
