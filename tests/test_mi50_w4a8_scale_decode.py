"""Exact register FP8 scale decode prototype; not enabled in the runtime backend."""
import importlib.util
from pathlib import Path
import sys

import unittest
from unittest.mock import patch
import torch

GPU = torch.cuda.is_available() and torch.version.hip and torch.cuda.get_device_properties(0).gcnArchName.startswith('gfx906')

@unittest.skipUnless(GPU, 'gfx906 required')
class ScaleDecodeTests(unittest.TestCase):
 def compare(self,bits,codebook,strided):
    from comfy_kitchen.backends.triton import w4a8_int8 as native
    source = Path(__file__).resolve().parents[1] / 'samples' / 'mi50_w4a8_scale_decode.py'
    name = 'comfy_kitchen.backends.triton.mi50_scale_decode_test'
    spec = importlib.util.spec_from_file_location(name, source)
    candidate = importlib.util.module_from_spec(spec)
    sys.modules[name] = candidate
    spec.loader.exec_module(candidate)
    with patch.dict('os.environ', {'MI50_W4A8_SCALE_DECODE':'1'}), torch.inference_mode():
        n, k = 67, 512
        q = torch.randint(-128, 128, (n, k * bits // 8), device='cuda', dtype=torch.int8)
        raw = torch.arange(n*k//16, device='cuda', dtype=torch.int64).remainder(256).to(torch.uint8)
        raw[(raw & 127) == 127] = 0
        scales = raw.reshape(n, k//16).view(torch.float8_e4m3fn)
        if strided:
            scales = scales.repeat_interleave(2, dim=1)[:, ::2]
        cb = torch.linspace(-3.125, 3.125, 16, device='cuda') if codebook else None
        reference = native._dequant_int4_grouped_to_int8(q, scales, cb, 16)
        result = candidate._dequant_int4_grouped_to_int8(q, scales, cb, 16)
        torch.cuda.synchronize()
        assert torch.equal(reference, result)

 def test_codebook_four_bit(self):self.compare(4,True,False)
 def test_uniform_four_bit_strided(self):self.compare(4,False,True)
 def test_uniform_six_bit(self):self.compare(6,False,False)

if __name__=='__main__':unittest.main()
