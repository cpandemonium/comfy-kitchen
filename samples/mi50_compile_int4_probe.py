"""Isolated MI50 experiments using trusted captures and an external INT4 kernel.

No custom-node import, global dispatch replacement, model conversion or deployment.
The external AGPL source stays outside this repository and is imported for research.
"""

import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import statistics
import sys
import time
import urllib.request

PHASE = sys.argv[1]
assert PHASE in ("compile", "compile-adapter", "int4", "int4-tuned")
ROOT = Path("/comfy/mnt/benchmarks/minimax-compile-int4-20261007")
VAE = Path("/comfy/mnt/benchmarks/minimax-vae-20261007")
AV = Path("/comfy/mnt/benchmarks/minimax-h3-20261007")
ROOT.mkdir(parents=True, exist_ok=True)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(ROOT / "inductor")
os.environ["TRITON_CACHE_DIR"] = str(ROOT / "triton")
os.environ["ROCM_INT8_KITCHEN_PATCH"] = "off"
sys.path[:0] = ["/opt/ComfyUI", "/opt/mi50"]
sys.argv = [__file__] + [
    arg for arg in shlex.split(os.environ["COMFYUI_ARGS"]) if arg != "--enable-manager"
]

import comfy.options

comfy.options.enable_args_parsing()
import torch
import triton
from launch import configure_runtime

configure_runtime()
from comfy_kitchen.backends.triton import quantization as ck
from comfy_kitchen.backends.triton.mi50_int8 import select_config
from comfy.ldm.modules.attention import optimized_attention


def idle():
    with urllib.request.urlopen("http://127.0.0.1:8283/queue", timeout=5) as response:
        queue = json.load(response)
    assert not queue["queue_running"] and not queue["queue_pending"], "UI job appeared"


def gpu(value):
    if isinstance(value, torch.Tensor):
        return value.cuda()
    if isinstance(value, dict):
        return {key: gpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(gpu(item) for item in value)
    return value


def measure(function):
    result = function()
    del result
    result = function()
    del result
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(9):
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = function()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - started)
        del result
    return dict(
        median_s=statistics.median(times),
        times_s=times,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
    )


def metrics(result, reference):
    changed = 0
    total = 0.0
    max_abs = 0.0
    finite = True
    for begin in range(0, reference.shape[0], 256):
        left = result[begin : begin + 256]
        right = reference[begin : begin + 256]
        finite = finite and bool(torch.isfinite(left).all())
        changed += int(torch.count_nonzero(left != right))
        delta = left.float() - right.float()
        max_abs = max(max_abs, float(delta.abs().max()))
        total += float(delta.square().sum())
    return dict(
        finite=finite,
        bitwise_equal=changed == 0,
        changed_values=changed,
        values=reference.numel(),
        max_abs=max_abs,
        rmse=(total / reference.numel()) ** 0.5,
    )


report = dict(
    phase=PHASE,
    torch=torch.__version__,
    hip=torch.version.hip,
    triton=triton.__version__,
    arch=torch.cuda.get_device_properties(0).gcnArchName,
    threads=torch.get_num_threads(),
    cpu_affinity=sorted(os.sched_getaffinity(0)),
    rows=[],
)


def save(row):
    report["rows"].append(row)
    (ROOT / (PHASE + ".json")).write_text(json.dumps(report, indent=2) + "\n")
    print("RESULT", json.dumps(row), flush=True)


def compile_one(name, function, arguments, reference, mode, fullgraph):
    idle()
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    row = dict(name=name, mode=mode, fullgraph=fullgraph)
    print("COMPILE_START", name, mode, fullgraph, flush=True)
    row["native"] = measure(lambda: function(*arguments))
    started = time.perf_counter()
    compiled = None
    try:
        compiled = torch.compile(function, mode=mode, fullgraph=fullgraph, dynamic=False)
        result = compiled(*arguments)
        torch.cuda.synchronize()
        row["first_call_s"] = time.perf_counter() - started
        row["numerical"] = metrics(result, reference)
        del result
        row["compiled"] = measure(lambda: compiled(*arguments))
        result = compiled(*arguments)
        row["repeat_numerical"] = metrics(result, reference)
        del result
        row["native_repeat"] = measure(lambda: function(*arguments))
        row["reduction_percent"] = (
            1 - row["compiled"]["median_s"] / row["native"]["median_s"]
        ) * 100
    except Exception as error:
        row["failed_after_s"] = time.perf_counter() - started
        row["error"] = type(error).__name__ + ": " + str(error)
    row["dynamo_counters"] = {
        str(group): {str(key): value for key, value in counts.items()}
        for group, counts in torch._dynamo.utils.counters.items()
    }
    save(row)
    compiled = None
    torch._dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()


def dense_vae_attention(q, k, v):
    batch, heads, tokens, dim = q.shape
    q = q.reshape(batch * heads, tokens, dim)
    k = k.reshape(batch * heads, tokens, dim)
    v = v.reshape(batch * heads, tokens, dim)
    scores = torch.bmm(q, k.transpose(-2, -1))
    probabilities = (scores * (dim**-0.5)).softmax(-1)
    result = torch.bmm(probabilities, v)
    return (
        result.reshape(batch, heads, tokens, dim)
        .permute(0, 2, 1, 3)
        .reshape(batch, tokens, heads * dim)
    )


def run_compile():
    implementation = ck.int8_linear
    if PHASE == "compile-adapter":
        # Obtain the raw JIT aliases before Dynamo, avoiding Autotuner.fn tracing.
        # Generate only a research copy from the exact installed Apache source.
        text = Path(ck.__file__).read_text()
        assert text.count("kernel.fn[grid](") == 1
        text = text.replace(
            "def int8_linear(\n",
            "_mi50_fixed_tensor_kernel = _int8_matmul_dequant_kernel.fn\n"
            "_mi50_fixed_row_kernel = _int8_matmul_dequant_per_row_kernel.fn\n\n"
            "def int8_linear(\n",
            1,
        ).replace(
            "kernel.fn[grid](",
            "(_mi50_fixed_row_kernel if is_per_channel else _mi50_fixed_tensor_kernel)[grid](",
            1,
        )
        path = ROOT / "quantization_compile_candidate.py"
        path.write_text(text)
        report["candidate_source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        spec = importlib.util.spec_from_file_location("mi50_compile_quantization_candidate", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        implementation = module.int8_linear
    paths = [VAE / "int8-3594x6144x2048.pt", VAE / "int8-7188x2048x8192.pt"]
    if PHASE == "compile-adapter":
        paths.append(AV / "int8-fp32-52708x5376x7168.pt")
    for path in paths:
        idle()
        name = path.name
        data = gpu(torch.load(path, map_location="cpu", weights_only=False))
        reference = data.pop("reference")
        args, kwargs = data["args"], data["kwargs"]

        def linear(x, weight, scale):
            return implementation(x, weight, scale, *args, **kwargs)

        arguments = (data["x"], data["weight"], data["weight_scale"])
        native = linear(*arguments)
        assert torch.equal(native, reference)
        del native
        compile_one(name, linear, arguments, reference, "default", True)
        if PHASE == "compile-adapter":
            compile_one(name, linear, arguments, reference, "reduce-overhead", True)
        else:
            compile_one(name, linear, arguments, reference, "default", False)
        del linear, data, arguments, reference
        args, kwargs = None, None
        gc.collect()
        torch.cuda.empty_cache()
    if PHASE == "compile-adapter":
        return
    for batch in (2, 4):
        idle()
        path = VAE / f"attention-{batch}.pt"
        data = gpu(torch.load(path, map_location="cpu", weights_only=False))
        arguments = tuple(data[key] for key in ("q", "k", "v"))
        native = optimized_attention(*arguments, data["heads"], **data["kwargs"])
        assert torch.equal(native, data["reference"])
        replay = dense_vae_attention(*arguments)
        assert torch.equal(replay, native)
        del replay, native
        for mode in ("default", "reduce-overhead"):
            compile_one(path.name, dense_vae_attention, arguments, data["reference"], mode, True)
        del data, arguments
        gc.collect()
        torch.cuda.empty_cache()


def pack(weight):
    low = weight[:, 0::2].to(torch.int16) & 15
    high = (weight[:, 1::2].to(torch.int16) & 15) << 4
    return (low | high).to(torch.int8).contiguous()


def unpack(packed):
    values = packed.to(torch.int16)
    low, high = values & 15, (values >> 4) & 15
    return (
        torch.stack(
            (torch.where(low >= 8, low - 16, low), torch.where(high >= 8, high - 16, high)),
            dim=-1,
        )
        .reshape(packed.shape[0], -1)
        .to(torch.int8)
    )


def prepared(data):
    kwargs = data["kwargs"]
    x = ck._apply_input_act(
        data["x"],
        kwargs.get("input_act"),
        kwargs.get("input_act_weight"),
        kwargs.get("input_act_eps", 0.0),
    )
    shape = x.shape
    x = x.reshape(-1, x.shape[-1])
    if kwargs.get("convrot", False):
        group = kwargs.get("convrot_groupsize", 256)
        h = ck._build_hadamard(group, device=x.device, dtype=x.dtype)
        x = ck._rotate_activation(x, h, group)
    return ck.triton_quantize_rowwise(x), shape


def run_int4():
    source = ROOT / "source" / "triton_int4_mm.py"
    report["external_commit"] = "a9560174bfe78cf16eb8fd21705c869d49eddf9d"
    report["external_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location("external_packed_int4", source)
    external = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(external)
    for m, n, k in ((17, 19, 2), (37, 29, 66), (64, 128, 256)):
        idle()
        torch.manual_seed(1234)
        a = torch.randint(-128, 128, (m, k), dtype=torch.int8, device="cuda")
        w = torch.randint(-8, 8, (n, k), dtype=torch.int8, device="cuda")
        packed = pack(w)
        assert torch.equal(unpack(packed), w)
        expected = (a.cpu().int() @ w.cpu().int().T).cuda()
        actual = external.triton_int4_mm(a, packed)
        assert torch.equal(actual, expected)
        save(dict(name="integer-smoke", shape=[m, n, k], bitwise_equal=True))
    del a, w, packed, actual, expected
    paths = [
        VAE / "int8-3594x6144x2048.pt",
        VAE / "int8-7188x2048x8192.pt",
        AV / "int8-fp32-52708x5376x7168.pt",
    ]
    for path in paths:
        idle()
        print("INT4_START", path.name, flush=True)
        data = gpu(torch.load(path, map_location="cpu", weights_only=False))
        reference = data.pop("reference")
        # Requantize actual ConvRot INT8 weights to symmetric per-channel INT4.
        weight = data["weight"]
        step = weight.float().abs().amax(dim=1, keepdim=True).clamp(min=1) / 7
        w4 = (weight.float() / step).round().clamp(-7, 7).to(torch.int8)
        scale = data["weight_scale"]
        scale = torch.as_tensor(scale, device=weight.device, dtype=torch.float32)
        scale4 = (scale.reshape(-1, 1) * step).reshape(-1)
        packed = pack(w4)
        assert torch.equal(unpack(packed), w4)
        (a, a_scale), output_shape = prepared(data)
        m, k = a.shape
        n = w4.shape[0]
        assert 127 * 7 * k < 2**24  # FP32 integer epilogue stays exact here.
        ones_a = torch.ones(m, device=a.device)
        ones_b = torch.ones(n, device=a.device)
        config = select_config(data["x"].reshape(-1, data["x"].shape[-1]), weight, data["x"].dtype)
        if config is None:
            config = dict(
                block_m=128, block_n=256, block_k=32, group_size_m=8, num_warps=8, num_stages=2
            )

        def expanded_mm():
            output = torch.empty((m, n), dtype=torch.int32, device=a.device)
            grid = lambda meta: (triton.cdiv(m, meta["block_m"]) * triton.cdiv(n, meta["block_n"]),)
            ck._int8_matmul_dequant_per_row_kernel.fn[grid](
                a_ptr=a,
                b_ptr=w4,
                c_ptr=output,
                a_scale_ptr=ones_a,
                b_scale_ptr=ones_b,
                bias_ptr=a,
                m=m,
                n=n,
                k=k,
                stride_am=a.stride(0),
                stride_ak=a.stride(1),
                stride_bk=w4.stride(1),
                stride_bn=w4.stride(0),
                stride_cm=output.stride(0),
                stride_cn=output.stride(1),
                has_bias=False,
                **config,
            )
            return output

        expanded = expanded_mm()
        result = external.triton_int4_mm(a, packed)
        assert torch.equal(result, expanded)
        del result
        variants = []
        if PHASE == "int4-tuned":
            for bm, bn, bk, warps, stages in (
                (64, 128, 32, 4, 2),
                (128, 256, 32, 8, 2),
                (128, 128, 64, 4, 2),
                (64, 256, 32, 4, 2),
                (128, 128, 32, 4, 2),
                (128, 256, 64, 8, 2),
            ):
                idle()
                external.BLOCK_SIZE_M = bm
                external.BLOCK_SIZE_N = bn
                external.BLOCK_SIZE_K = bk
                external.NUM_WARPS = warps
                external.NUM_STAGES = stages
                variant = dict(block_m=bm, block_n=bn, block_k=bk, warps=warps, stages=stages)
                print("INT4_CONFIG", path.name, variant, flush=True)
                try:
                    result = external.triton_int4_mm(a, packed)
                    variant["integer_equal"] = torch.equal(result, expanded)
                    del result
                    assert variant["integer_equal"]
                    variant.update(measure(lambda: external.triton_int4_mm(a, packed)))
                except Exception as error:
                    variant["error"] = type(error).__name__ + ": " + str(error)
                variants.append(variant)
            winner = min(
                (variant for variant in variants if "median_s" in variant),
                key=lambda variant: variant["median_s"],
            )
            external.BLOCK_SIZE_M = winner["block_m"]
            external.BLOCK_SIZE_N = winner["block_n"]
            external.BLOCK_SIZE_K = winner["block_k"]
            external.NUM_WARPS = winner["warps"]
            external.NUM_STAGES = winner["stages"]
        del expanded
        args, kwargs = data["args"], data["kwargs"]

        def packed_linear():
            (activation, activation_scale), shape = prepared(data)
            output = external.triton_int4_mm(activation, packed).float()
            output *= activation_scale.reshape(-1, 1) * scale4.reshape(1, -1)
            bias = kwargs.get("bias", args[0] if args else None)
            if bias is not None:
                output += bias.float()
            output = output.to(kwargs.get("out_dtype", torch.bfloat16)).reshape(*shape[:-1], n)
            return ck._apply_residual(output, kwargs.get("residual"), kwargs.get("residual_scale"))

        same4 = ck.int8_linear(data["x"], w4, scale4, *args, **kwargs)
        result = packed_linear()
        row = dict(
            name=path.name,
            shape=[m, n, k],
            weight_bytes_int8=weight.numel(),
            weight_bytes_packed=packed.numel(),
            integer_gemm_equal=True,
            packed_vs_same4=metrics(result, same4),
            int4_vs_original_int8=metrics(result, reference),
            packed_config=dict(
                block_m=external.BLOCK_SIZE_M,
                block_n=external.BLOCK_SIZE_N,
                block_k=external.BLOCK_SIZE_K,
                warps=external.NUM_WARPS,
                stages=external.NUM_STAGES,
            ),
            variants=variants,
        )
        del result, same4
        row["expanded_mm"] = measure(expanded_mm)
        row["packed_mm"] = measure(lambda: external.triton_int4_mm(a, packed))
        row["native_int8_linear"] = measure(
            lambda: ck.int8_linear(data["x"], weight, data["weight_scale"], *args, **kwargs)
        )
        row["native_same4_linear"] = measure(
            lambda: ck.int8_linear(data["x"], w4, scale4, *args, **kwargs)
        )
        row["packed_linear"] = measure(packed_linear)
        row["native_repeat"] = measure(
            lambda: ck.int8_linear(data["x"], weight, data["weight_scale"], *args, **kwargs)
        )
        save(row)
        del expanded_mm, packed_linear, reference, a_scale, scale, step
        data = weight = w4 = packed = scale4 = a = ones_a = ones_b = args = kwargs = None
        gc.collect()
        torch.cuda.empty_cache()


idle()
with torch.inference_mode():
    (run_compile if PHASE.startswith("compile") else run_int4)()
idle()
