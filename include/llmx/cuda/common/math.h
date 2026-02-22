#pragma once

#include <cuda.h>
#include <cute/config.hpp>

namespace llmx {

CUTE_HOST_DEVICE constexpr int clz(int x) {
  for (int i = 31; i >= 0; --i) {
    if ((1 << i) & x) {
      return int(31 - i);
    }
  }
  return int(32);
}

CUTE_HOST_DEVICE constexpr bool is_pow2(int x) { return (x & (x - 1)) == 0; }

CUTE_HOST_DEVICE constexpr int log2(int x) {
  int a = int(31 - clz(x));
  // add 1 if not a power of 2
  if (!is_pow2(x)) {
    a += 1;
  }
  return a;
}

// wrapper of PTX ex2.approx instruction, which computes 2^x
CUTE_HOST_DEVICE float exp2(float x) {
#if defined(__CUDA_ARCH__)
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
#else
  return std::exp2(x);
#endif
}

// wrapper of PTX rcp.approx instruction, which computes 1/x
CUTE_HOST_DEVICE float rcp(float x) {
#if defined(__CUDA_ARCH__)
  float y;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
#else
  return 1.0f / x;
#endif
}

// wrapper of PTX tanh.approx instruction, which computes tanh(x)
CUTE_HOST_DEVICE float tanh(float x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
  float y;
  asm volatile("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
#else
  return std::tanh(x);
#endif
}

} // namespace llmx