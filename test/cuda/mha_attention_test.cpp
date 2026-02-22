#include "llmx/cuda/attention/attention_api.h"
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

using namespace llmx;

int main() {
  constexpr int n_heads = 4;
  constexpr int n_kv_heads = 2;
  constexpr int seq_len = 256;
  constexpr int head_dim = 128;
  const float sm_scale = 1.0f / sqrtf(static_cast<float>(head_dim));

  Tensor q({seq_len, n_heads, head_dim}, ScalarType::kFP16, cuda0);
  Tensor k({seq_len, n_kv_heads, head_dim}, ScalarType::kFP16, cuda0);
  Tensor v({seq_len, n_kv_heads, head_dim}, ScalarType::kFP16, cuda0);
  Tensor output({seq_len, n_heads, head_dim}, ScalarType::kFP16, cuda0);

  q.allocate();
  k.allocate();
  v.allocate();
  output.allocate();

  llmx_single_mha_attention(output, q, k, v, std::nullopt, sm_scale);

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    return 1;
  }

  printf("done\n");
  return 0;
}
