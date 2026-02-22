#pragma once

#include <cstddef>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace llmx {

template <typename Src, typename Dst>
void launch_scalar_convert(const Src *src, Dst *dst, size_t num);

template <typename SrcType, typename DstType> struct ScalarConverter;
template <typename T> struct ScalarConverter<T, T> {
  void operator()(const T *src, T *dst, size_t num) {}
};
template <> struct ScalarConverter<float, __half> {
  void operator()(const float *src, __half *dst, size_t num) {
    launch_scalar_convert(src, dst, num);
  }
};
template <> struct ScalarConverter<float, __nv_bfloat16> {
  void operator()(const float *src, __nv_bfloat16 *dst, size_t num) {
    launch_scalar_convert(src, dst, num);
  }
};
template <> struct ScalarConverter<__half, float> {
  void operator()(const __half *src, float *dst, size_t num) {
    launch_scalar_convert(src, dst, num);
  }
};
template <> struct ScalarConverter<__half, __nv_bfloat16> {
  void operator()(const __half *src, __nv_bfloat16 *dst, size_t num) {
    launch_scalar_convert(src, dst, num);
  }
};
template <> struct ScalarConverter<__nv_bfloat16, float> {
  void operator()(const __nv_bfloat16 *src, float *dst, size_t num) {
    launch_scalar_convert(src, dst, num);
  }
};
template <> struct ScalarConverter<__nv_bfloat16, __half> {
  void operator()(const __nv_bfloat16 *src, __half *dst, size_t num) {
    launch_scalar_convert(src, dst, num);
  }
};

} // namespace llmx