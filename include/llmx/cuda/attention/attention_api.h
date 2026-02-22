#pragma once

#include "llmx/tensor.h"
#include <optional>

namespace llmx {

void llmx_single_mha_attention(
    Tensor &output,                            // [q_len, n_heads, head_dim]
    const Tensor &q,                           // [q_len, n_heads, head_dim]
    const Tensor &k,                           // [kv_len, n_kv_heads, head_dim]
    const Tensor &v,                           // [kv_len, n_kv_heads, head_dim]
    const std::optional<Tensor> &alibi_slopes, // [n_heads]
    float sm_scale, int sliding_window = -1, float logits_soft_cap = 0);

} // namespace llmx