#include "llmx/scalar.h"

namespace llmx {

template <typename Src, typename Dst> struct ScalarCast;
template <> struct ScalarCast<float, __half> {
  __device__ __half operator()(float x) const { return __float2half(x); }
};
template <> struct ScalarCast<float, __nv_bfloat16> {
  __device__ __nv_bfloat16 operator()(float x) const {
    return __float2bfloat16(x);
  }
};
template <> struct ScalarCast<__half, float> {
  __device__ float operator()(__half x) const { return __half2float(x); }
};
template <> struct ScalarCast<__half, __nv_bfloat16> {
  __device__ __nv_bfloat16 operator()(__half x) const {
    return __float2bfloat16(__half2float(x));
  }
};
template <> struct ScalarCast<__nv_bfloat16, float> {
  __device__ float operator()(__nv_bfloat16 x) const {
    return __bfloat162float(x);
  }
};
template <> struct ScalarCast<__nv_bfloat16, __half> {
  __device__ __half operator()(__nv_bfloat16 x) const {
    return __float2half(__bfloat162float(x));
  }
};

template <typename Src, typename Dst>
__global__ void scalar_convert_kernel(const Src *src, Dst *dst, size_t num) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num) {
    dst[idx] = ScalarCast<Src, Dst>{}(src[idx]);
  }
}

template <typename Src, typename Dst>
void launch_scalar_convert(const Src *src, Dst *dst, size_t num) {
  constexpr int kBlockSize = 256;
  const int grid_size = static_cast<int>((num + kBlockSize - 1) / kBlockSize);
  scalar_convert_kernel<Src, Dst><<<grid_size, kBlockSize>>>(src, dst, num);
}

template void launch_scalar_convert<float, __half>(const float *, __half *,
                                                   size_t);
template void launch_scalar_convert<float, __nv_bfloat16>(const float *,
                                                          __nv_bfloat16 *,
                                                          size_t);
template void launch_scalar_convert<__half, float>(const __half *, float *,
                                                   size_t);
template void launch_scalar_convert<__half, __nv_bfloat16>(const __half *,
                                                           __nv_bfloat16 *,
                                                           size_t);
template void launch_scalar_convert<__nv_bfloat16, float>(const __nv_bfloat16 *,
                                                          float *, size_t);
template void
launch_scalar_convert<__nv_bfloat16, __half>(const __nv_bfloat16 *, __half *,
                                             size_t);

} // namespace llmx