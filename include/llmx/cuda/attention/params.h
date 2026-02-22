#pragma once

#include "cute/layout.hpp"

namespace llmx {

struct MHAParams {
  void *__restrict__ o_ptr = nullptr;
  const void *__restrict__ q_ptr = nullptr;
  const void *__restrict__ k_ptr = nullptr;
  const void *__restrict__ v_ptr = nullptr;

  int batch_size;
  int q_len;
  int kv_len;
  int n_heads;
  int n_kv_heads;
  int head_dim;

  const float *__restrict__ alibi_slopes_ptr = nullptr;

  float sm_scale;
  int sliding_window;
  float logits_soft_cap;
  int group_size;

  // [batch_size, seq_len, n_heads, head_dim]
  using Stride = cute::Stride<int, int, int, cute::_1>;

  Stride q_stride;
  Stride k_stride;
  Stride v_stride;
  Stride o_stride;
};

} // namespace llmx