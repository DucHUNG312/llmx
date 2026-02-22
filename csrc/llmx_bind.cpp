#include "llmx/cuda/attention/attention_api.h"
#include "llmx/tensor.h"
#include <torch/extension.h>

namespace py = pybind11;

namespace {

llmx::ScalarType to_llmx_dtype(torch::ScalarType t) {
  switch (t) {
  case torch::kHalf:
    return llmx::ScalarType::kFP16;
  case torch::kBFloat16:
    return llmx::ScalarType::kBF16;
  case torch::kFloat:
    return llmx::ScalarType::kFP32;
  default:
    TORCH_CHECK(false, "llmx: unsupported dtype ", t,
                ". Only float16, bfloat16, float32 are supported.");
  }
}

llmx::Tensor tensor_from_torch(const torch::Tensor &src) {
  TORCH_CHECK(src.is_cuda(), "Expected a CUDA tensor");
  TORCH_CHECK(src.is_contiguous(), "Expected a contiguous tensor");

  std::vector<int> dims(src.sizes().begin(), src.sizes().end());
  llmx::ScalarType dtype = to_llmx_dtype(src.scalar_type());
  llmx::Device device(llmx::DeviceType::kCUDA, src.device().index());
  return llmx::Tensor(src.data_ptr(), dims, dtype, device);
}

} // namespace

void single_mha_attention_half(
    torch::Tensor output, // [q_len,  n_heads, head_dim]
    torch::Tensor q,      // [q_len,  n_heads,    head_dim]
    torch::Tensor k,      // [kv_len, n_kv_heads, head_dim]
    torch::Tensor v,      // [kv_len, n_kv_heads, head_dim]
    float sm_scale, int sliding_window = -1, float logits_soft_cap = 0.0f) {
  TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3,
              "q, k, v must be 3-D: [seq, heads, head_dim]");
  TORCH_CHECK(q.scalar_type() == k.scalar_type() &&
                  q.scalar_type() == v.scalar_type(),
              "q, k, v must have the same dtype");
  TORCH_CHECK(q.scalar_type() == torch::kHalf ||
                  q.scalar_type() == torch::kBFloat16,
              "Only float16 and bfloat16 are supported");
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(),
              "q, k, v must be CUDA tensors");
  TORCH_CHECK(q.device() == k.device() && q.device() == v.device(),
              "q, k, v must be on the same device");

  const int n_heads = q.size(1), n_kv_heads = k.size(1);
  TORCH_CHECK(n_heads % n_kv_heads == 0, "n_heads (", n_heads,
              ") must be divisible by n_kv_heads (", n_kv_heads, ")");
  TORCH_CHECK(q.size(2) == k.size(2) && q.size(2) == v.size(2),
              "head_dim must match across q, k, v");

  q = q.contiguous();
  k = k.contiguous();
  v = v.contiguous();

  llmx::Tensor llmx_q = tensor_from_torch(q);
  llmx::Tensor llmx_k = tensor_from_torch(k);
  llmx::Tensor llmx_v = tensor_from_torch(v);

  llmx::Tensor llmx_out = tensor_from_torch(output);
  llmx::llmx_single_mha_attention(llmx_out, llmx_q, llmx_k, llmx_v,
                                  std::nullopt, sm_scale, sliding_window,
                                  logits_soft_cap);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("single_mha_attention_half", &single_mha_attention_half,
        py::arg("output"), py::arg("q"), py::arg("k"), py::arg("v"),
        py::arg("sm_scale"), py::arg("sliding_window") = -1,
        py::arg("logits_soft_cap") = 0.0f);
}