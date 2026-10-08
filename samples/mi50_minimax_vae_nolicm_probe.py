"""One scheduling candidate on two dominant real FP16 VAE fc1 forms."""

import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path("/comfy/mnt/benchmarks/minimax-memory-query32-20261008")
OLD = Path("/comfy/mnt/benchmarks/minimax-vae-20261007")
sys.path[:0] = ["/opt/ComfyUI", "/opt/mi50"]
import torch  # noqa: E402
from launch import configure_runtime  # noqa: E402

configure_runtime()
from comfy_kitchen.backends.triton import quantization as base, mi50_int8  # noqa: E402


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


scope = module("vae_nolicm_scope", ROOT / "mi50_int8.py")
candidate = module("vae_nolicm_quantization", ROOT / "quantization.py")
mi50_int8.use_chunked_convrot = scope.use_chunked_convrot
mi50_int8.use_nolicm_vae_gemm = scope.use_nolicm_vae_gemm
os.environ["MI50_MINIMAX_VAE_GEMM_NO_LICM"] = "1"
report = {
    "candidate": "disable LICM in two native INT8 GEMM loop variants; same tiles/arithmetic",
    "policy": "One warmup and one timed full linear per arm per shape.",
    "forms": [],
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


def write():
    (ROOT / "vae-nolicm-probe.json").write_text(json.dumps(report, indent=2) + "\n")


with torch.inference_mode():
    for m in (7188, 3594):
        data = gpu(
            torch.load(OLD / f"int8-{m}x16384x2048.pt", map_location="cpu", weights_only=False)
        )
        golden = data.pop("reference").cpu()
        row = {"shape": [m, 16384, 2048], "arms": []}
        for label, mod in [("baseline", base), ("candidate", candidate)]:

            def call(_mod=mod, _data=data):
                return _mod.int8_linear(
                    _data["x"],
                    _data["weight"],
                    _data["weight_scale"],
                    *_data["args"],
                    **_data["kwargs"],
                )

            idle()
            out = call()
            torch.cuda.synchronize()
            del out
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            started = time.perf_counter()
            a.record()
            out = call()
            b.record()
            b.synchronize()
            arm = {
                "name": label,
                "gpu_s": a.elapsed_time(b) / 1000,
                "wall_s": time.perf_counter() - started,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "bitwise_equal": torch.equal(out.cpu(), golden),
                "finite": bool(torch.isfinite(out).all()),
            }
            row["arms"].append(arm)
            del out
            print("VAE_LINEAR", json.dumps(dict(shape=row["shape"], **arm)), flush=True)
        row["gain_percent"] = 100 * (1 - row["arms"][1]["gpu_s"] / row["arms"][0]["gpu_s"])
        report["forms"].append(row)
        write()
        del data, golden
        gc.collect()
        torch.cuda.empty_cache()
    report["accepted_for_full_decode"] = all(
        a["bitwise_equal"] and a["finite"] for row in report["forms"] for a in row["arms"]
    ) and sum(row["arms"][1]["gpu_s"] for row in report["forms"]) < 0.98 * sum(
        row["arms"][0]["gpu_s"] for row in report["forms"]
    )
    report["complete"] = True
    write()
    print("RESULT", report["accepted_for_full_decode"], flush=True)
