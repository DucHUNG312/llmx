#include "llmx/cuda/attention/attention_api.h"
#include "cute/layout.hpp"
#include "llmx/cuda/attention/params.h"
#include "llmx/dispatch.h"
#include "llmx/utility.h"

namespace llmx {

template <typename ELEMENT, const int HEAD_DIM, const bool EVEN_K,
          const bool ALIBI, const bool SOFT_CAP, const bool LOCAL>
void single_mha_attention_launcher(const MHAParams &params,
                                   cudaStream_t stream = 0);

void llmx_single_mha_attention(
    Tensor &output,                            // [q_len, n_heads, head_dim]
    const Tensor &q,                           // [q_len, n_heads, head_dim]
    const Tensor &k,                           // [kv_len, n_kv_heads, head_dim]
    const Tensor &v,                           // [kv_len, n_kv_heads, head_dim]
    const std::optional<Tensor> &alibi_slopes, // [n_heads]
    float sm_scale, int sliding_window, float logits_soft_cap) {

  LLMX_CHECK(output.contiguous());
  LLMX_CHECK(q.contiguous());
  LLMX_CHECK(k.contiguous());
  LLMX_CHECK(v.contiguous());

  const auto head_dim = q.dim(2);
  const auto n_heads = q.dim(1);
  const auto q_len = q.dim(0);
  const auto n_kv_heads = k.dim(1);
  const auto kv_len = k.dim(0);

  MHAParams params;
  params.o_ptr = output.data_ptr();
  params.q_ptr = q.const_data_ptr();
  params.k_ptr = k.const_data_ptr();
  params.v_ptr = v.const_data_ptr();
  params.alibi_slopes_ptr = alibi_slopes.has_value()
                                ? alibi_slopes->const_data_ptr<float>()
                                : nullptr;
  params.batch_size = 1;
  params.q_len = q_len;
  params.kv_len = kv_len;
  params.n_heads = n_heads;
  params.n_kv_heads = n_kv_heads;
  params.sm_scale = sm_scale;
  params.sliding_window = sliding_window;
  params.logits_soft_cap = logits_soft_cap;
  params.q_stride = cute::make_stride(0, q.stride(0), q.stride(1), cute::_1{});
  params.o_stride =
      cute::make_stride(0, output.stride(0), output.stride(1), cute::_1{});
  params.k_stride = cute::make_stride(0, k.stride(0), k.stride(1), cute::_1{});
  params.v_stride = cute::make_stride(0, v.stride(0), v.stride(1), cute::_1{});

  cudaStream_t stream = 0;

  LLMX_DISPATCH_CUTE_DTYPE(q.dtype(), ELEMENT, [&] {
    LLMX_DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, [&] {
      LLMX_DISPATCH_BOOL(head_dim == HEAD_DIM, EVEN_K, [&] {
        LLMX_DISPATCH_BOOL(params.alibi_slopes_ptr != nullptr, ALIBI, [&] {
          LLMX_DISPATCH_BOOL(logits_soft_cap > 0, SOFT_CAP, [&] {
            LLMX_DISPATCH_BOOL(sliding_window >= 0, LOCAL, [&] {
              params.normalize();
              single_mha_attention_launcher<ELEMENT, HEAD_DIM, EVEN_K, ALIBI,
                                            SOFT_CAP, LOCAL>(params, stream);
            });
          });
        });
      });
    });
  });
}

} // namespace llmx