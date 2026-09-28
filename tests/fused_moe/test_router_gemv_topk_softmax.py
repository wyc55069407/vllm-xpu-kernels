# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import vllm_xpu_kernels._moe_C  # noqa: F401

DEVICE = "xpu"


def _reference(x, w, top_k, renormalize):
    logits = torch.nn.functional.linear(x, w)
    m = x.shape[0]
    topk_weights = torch.empty(m, top_k, dtype=torch.float32, device=DEVICE)
    topk_ids = torch.empty(m, top_k, dtype=torch.int32, device=DEVICE)
    token_expert_indices = torch.empty_like(topk_ids)
    torch.ops._moe_C.topk_softmax(topk_weights, topk_ids, token_expert_indices,
                                  logits, renormalize, None, None)
    return logits, topk_weights, topk_ids


@pytest.mark.parametrize("m", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("e,k,top_k", [(256, 2048, 8), (128, 2816, 8),
                                       (512, 1024, 10), (64, 4096, 4)])
@pytest.mark.parametrize("renormalize", [True, False])
def test_router_gemv_topk_softmax(m, e, k, top_k, renormalize):
    torch.manual_seed(0)
    x = torch.randn(m, k, device=DEVICE, dtype=torch.float16)
    w = torch.randn(e, k, device=DEVICE, dtype=torch.float16) * 0.02
    logits = torch.empty(m, e, device=DEVICE, dtype=torch.float32)
    topk_weights = torch.empty(m, top_k, device=DEVICE, dtype=torch.float32)
    topk_ids = torch.empty(m, top_k, device=DEVICE, dtype=torch.int32)
    torch.ops._moe_C.router_gemv_topk_softmax(x, w, logits, topk_weights,
                                              topk_ids, renormalize)
    ref_logits, ref_w, ref_ids = _reference(x, w, top_k, renormalize)
    torch.testing.assert_close(logits,
                               ref_logits.float(),
                               atol=2e-3,
                               rtol=2e-3)
    # Compare as sets per token (order of equal-probability picks may
    # differ), then weights by expert id.
    ids, order = topk_ids.long().sort(dim=-1)
    rids, rorder = ref_ids.long().sort(dim=-1)
    assert torch.equal(ids, rids)
    torch.testing.assert_close(topk_weights.gather(-1, order),
                               ref_w.gather(-1, rorder),
                               atol=1e-3,
                               rtol=1e-3)


def test_router_gemv_topk_softmax_strided_input():
    torch.manual_seed(0)
    base = torch.randn(4, 4096, device=DEVICE, dtype=torch.float16)
    x = base[:, :2048]
    w = torch.randn(256, 2048, device=DEVICE, dtype=torch.float16) * 0.02
    logits = torch.empty(4, 256, device=DEVICE, dtype=torch.float32)
    topk_weights = torch.empty(4, 8, device=DEVICE, dtype=torch.float32)
    topk_ids = torch.empty(4, 8, device=DEVICE, dtype=torch.int32)
    torch.ops._moe_C.router_gemv_topk_softmax(x, w, logits, topk_weights,
                                              topk_ids, True)
    _, ref_w, ref_ids = _reference(x.contiguous(), w, 8, True)
    assert torch.equal(topk_ids.sort(-1).values, ref_ids.sort(-1).values)
