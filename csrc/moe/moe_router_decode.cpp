#include <sycl/sycl.hpp>

#include <ATen/DeviceGuard.h>
#include <cmath>
#include <cstdint>

#include "../utils.h"
#include "moe_ops.h"

namespace vllm {
namespace moe {

// Router for small-M decode: logits = x @ W^T (fp16 weight, fp32 accumulate,
// rounded to fp16 like an fp16 F.linear), then softmax top-k with optional
// renormalization (same result as F.linear + topk_softmax).
//
// Kernel 1 (router_gemv): one 16-lane sub-group per expert row computes the
//   logits of that expert for all M tokens (x is read through the cache).
// Kernel 2 (router_topk_softmax): one sub-group per token; each lane holds
//   E / 16 logits, selects the top-k by repeated sub-group arg-max (lowest
//   expert id wins ties) and writes softmax probabilities.
constexpr int kRouterSg = 16;
constexpr int kRouterSgPerWg = 4;
constexpr int kRouterMaxM = 8;
constexpr int kRouterMaxEpl = 32;  // up to 512 experts

class router_gemv_kernel {
 public:
  router_gemv_kernel(
      const sycl::half* x,
      int64_t x_stride,
      const sycl::half* w,
      float* logits,
      int M,
      int E,
      int K)
      : x(x), x_stride(x_stride), w(w), logits(logits), M(M), E(E), K(K) {}

  [[sycl::reqd_sub_group_size(kRouterSg)]] void
  operator()(sycl::nd_item<1> item) const {
    auto sg = item.get_sub_group();
    const int lane = sg.get_local_linear_id();
    const int e = item.get_group(0) * kRouterSgPerWg + sg.get_group_linear_id();
    if (e >= E) return;

    float acc[kRouterMaxM];
#pragma unroll
    for (int m = 0; m < kRouterMaxM; ++m) acc[m] = 0.f;

    using vec8 = sycl::vec<sycl::half, 8>;
    const sycl::half* wr = w + static_cast<int64_t>(e) * K;
    for (int k = lane * 8; k < K; k += kRouterSg * 8) {
      const vec8 wv = *reinterpret_cast<const vec8*>(wr + k);
#pragma unroll
      for (int m = 0; m < kRouterMaxM; ++m) {
        if (m < M) {
          const vec8 xv = *reinterpret_cast<const vec8*>(x + m * x_stride + k);
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            acc[m] += static_cast<float>(wv[j]) * static_cast<float>(xv[j]);
          }
        }
      }
    }
#pragma unroll
    for (int m = 0; m < kRouterMaxM; ++m) {
      if (m < M) {
        const float s = sycl::reduce_over_group(sg, acc[m], sycl::plus<float>());
        if (lane == 0) {
          logits[m * E + e] = static_cast<float>(static_cast<sycl::half>(s));
        }
      }
    }
  }

 private:
  const sycl::half* x;
  int64_t x_stride;
  const sycl::half* w;
  float* logits;
  int M, E, K;
};

class router_topk_softmax_kernel {
 public:
  router_topk_softmax_kernel(
      const float* logits,
      float* topk_weights,
      int32_t* topk_ids,
      int E,
      int top_k,
      bool renormalize)
      : logits(logits),
        topk_weights(topk_weights),
        topk_ids(topk_ids),
        E(E),
        top_k(top_k),
        renormalize(renormalize) {}

  [[sycl::reqd_sub_group_size(kRouterSg)]] void
  operator()(sycl::nd_item<1> item) const {
    auto sg = item.get_sub_group();
    const int lane = sg.get_local_linear_id();
    const int t = item.get_group(0);
    const float* l = logits + static_cast<int64_t>(t) * E;
    const int epl = (E + kRouterSg - 1) / kRouterSg;

    // Expert e = lane + i * 16 lives in slot i of that lane.
    float v[kRouterMaxEpl];
    float mx = -INFINITY;
#pragma unroll
    for (int i = 0; i < kRouterMaxEpl; ++i) {
      const int e = lane + i * kRouterSg;
      v[i] = (i < epl && e < E) ? l[e] : -INFINITY;
      mx = sycl::fmax(mx, v[i]);
    }
    mx = sycl::reduce_over_group(sg, mx, sycl::maximum<float>());
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < kRouterMaxEpl; ++i) {
      if (i < epl) sum += sycl::exp(v[i] - mx);
    }
    sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>());

    float sel_sum = 0.f;
    float sel_p = 0.f;
    int sel_e = 0;
    for (int j = 0; j < top_k; ++j) {
      // Local best (value, lowest expert id), then sub-group arg-max.
      float bv = -INFINITY;
      int be = E;
#pragma unroll
      for (int i = 0; i < kRouterMaxEpl; ++i) {
        const int e = lane + i * kRouterSg;
        if (i < epl && e < E && (v[i] > bv || (v[i] == bv && e < be))) {
          bv = v[i];
          be = e;
        }
      }
      const float gv = sycl::reduce_over_group(sg, bv, sycl::maximum<float>());
      const int ge = sycl::reduce_over_group(
          sg, bv == gv ? be : E, sycl::minimum<int>());
      const float p = sycl::exp(gv - mx) / sum;
      sel_sum += p;
      if (lane == j) {
        sel_p = p;
        sel_e = ge;
      }
      if (ge % kRouterSg == lane) {
        const int slot = ge / kRouterSg;
#pragma unroll
        for (int i = 0; i < kRouterMaxEpl; ++i) {
          if (i == slot) v[i] = -INFINITY;
        }
      }
    }
    if (lane < top_k) {
      const float scale = renormalize ? 1.f / sel_sum : 1.f;
      topk_weights[t * top_k + lane] = sel_p * scale;
      topk_ids[t * top_k + lane] = sel_e;
    }
  }

 private:
  const float* logits;
  float* topk_weights;
  int32_t* topk_ids;
  int E, top_k;
  bool renormalize;
};

}  // namespace moe
}  // namespace vllm

void router_gemv_topk_softmax(
    const torch::Tensor& x,              // [M, K] fp16
    const torch::Tensor& router_weight,  // [E, K] fp16
    torch::Tensor& logits,               // [M, E] fp32 (scratch / output)
    torch::Tensor& topk_weights,         // [M, top_k] fp32
    torch::Tensor& topk_ids,             // [M, top_k] int32
    bool renormalize) {
  using namespace vllm::moe;
  const at::DeviceGuard device_guard(x.device());
  TORCH_CHECK(
      x.dim() == 2 && x.scalar_type() == at::kHalf && x.stride(1) == 1,
      "router_gemv_topk_softmax: x must be fp16 [M, K] with unit inner "
      "stride");
  TORCH_CHECK(
      router_weight.dim() == 2 && router_weight.scalar_type() == at::kHalf &&
          router_weight.is_contiguous(),
      "router_gemv_topk_softmax: router_weight must be contiguous fp16 [E, K]");
  const int64_t M = x.size(0);
  const int64_t K = x.size(1);
  const int64_t E = router_weight.size(0);
  const int64_t top_k = topk_ids.size(-1);
  TORCH_CHECK(router_weight.size(1) == K, "router_gemv_topk_softmax: K mismatch");
  TORCH_CHECK(
      M <= kRouterMaxM, "router_gemv_topk_softmax: at most 8 tokens");
  TORCH_CHECK(
      K % (kRouterSg * 8) == 0,
      "router_gemv_topk_softmax: K must be a multiple of 128");
  TORCH_CHECK(
      E <= kRouterSg * kRouterMaxEpl && top_k >= 1 && top_k <= kRouterSg &&
          top_k <= E,
      "router_gemv_topk_softmax: unsupported num_experts / top_k");
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(router_weight.data_ptr()) % 16 == 0 &&
          reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
          x.stride(0) % 8 == 0,
      "router_gemv_topk_softmax: x and router_weight must be 16-byte "
      "aligned");
  TORCH_CHECK(
      logits.scalar_type() == at::kFloat && logits.is_contiguous() &&
      logits.numel() >= M * E);
  TORCH_CHECK(
      topk_weights.scalar_type() == at::kFloat &&
      topk_weights.is_contiguous() && topk_weights.numel() == M * top_k);
  TORCH_CHECK(
      topk_ids.scalar_type() == at::kInt && topk_ids.is_contiguous() &&
      topk_ids.numel() == M * top_k);
  if (M == 0) return;

  auto& queue = vllm::xpu::vllmGetQueue();
  const auto* xp = reinterpret_cast<const sycl::half*>(x.data_ptr());
  const auto* wp = reinterpret_cast<const sycl::half*>(router_weight.data_ptr());
  float* lp = logits.data_ptr<float>();
  const int64_t num_wg = (E + kRouterSgPerWg - 1) / kRouterSgPerWg;
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(
            num_wg * kRouterSgPerWg * kRouterSg, kRouterSgPerWg * kRouterSg),
        router_gemv_kernel(
            xp, x.stride(0), wp, lp, static_cast<int>(M), static_cast<int>(E),
            static_cast<int>(K)));
  });
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(M * kRouterSg, kRouterSg),
        router_topk_softmax_kernel(
            lp, topk_weights.data_ptr<float>(), topk_ids.data_ptr<int32_t>(),
            static_cast<int>(E), static_cast<int>(top_k), renormalize));
  });
}
