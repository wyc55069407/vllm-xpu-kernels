# SPDX-License-Identifier: Apache-2.0
# fp8_gemm_w8a16 single-row (decode) inputs take the SYCL GEMV path on BMG;
# the result must match the fp32 reference like the oneDNN path does.
import pytest
import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401

DEVICE = "xpu"


def _reference(x, w, scale):
    return (x.float() @ (w.float() * scale).t()).to(x.dtype)


@pytest.mark.parametrize("n,k", [(4608, 2048), (2048, 2048), (6144, 2048),
                                 (32, 2048), (1000, 320), (8, 64)])
@pytest.mark.parametrize("m", [1, 2, 8])
@pytest.mark.parametrize("x_3d", [False, True])
def test_fp8_gemm_w8a16_small_m(n, k, m, x_3d):
    torch.manual_seed(0)
    w = (torch.randn(n, k, device=DEVICE) * 0.05).to(torch.float8_e4m3fn)
    scale = torch.tensor([0.02], device=DEVICE)
    x = torch.randn(m, k, device=DEVICE, dtype=torch.float16)
    if x_3d:
        x = x.view(m, 1, k)
    out = torch.ops._xpu_C.fp8_gemm_w8a16(x, w.t(), scale, None)
    ref = _reference(x.reshape(m, k), w, scale).reshape(*x.shape[:-1], n)
    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, atol=2e-3, rtol=2e-2)


def test_fp8_gemm_w8a16_strided_row():
    torch.manual_seed(0)
    w = (torch.randn(2048, 2048, device=DEVICE) * 0.05).to(torch.float8_e4m3fn)
    scale = torch.tensor([0.02], device=DEVICE)
    base = torch.randn(1, 4096, device=DEVICE, dtype=torch.float16)
    x = base[:, 2048:]
    out = torch.ops._xpu_C.fp8_gemm_w8a16(x, w.t(), scale, None)
    torch.testing.assert_close(out, _reference(x.contiguous(), w, scale),
                               atol=2e-3, rtol=2e-2)
