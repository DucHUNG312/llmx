#pragma once

#define LLMX_DISPATCH_SCALAR_TYPE(TYPE, NAME, ...)                             \
  [&] {                                                                        \
    if (TYPE == ScalarType::kFP32) {                                           \
      using NAME = float;                                                      \
      return __VA_ARGS__();                                                    \
    } else if (TYPE == ScalarType::kFP16) {                                    \
      using NAME = __half;                                                     \
      return __VA_ARGS__();                                                    \
    } else if (TYPE == ScalarType::kBF16) {                                    \
      using NAME = __nv_bfloat16;                                              \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()

#define LLMX_DISPATCH_BOOL(VALUE, NAME, ...)                                   \
  [&] {                                                                        \
    if (VALUE) {                                                               \
      constexpr static bool NAME = true;                                       \
      return __VA_ARGS__();                                                    \
    } else {                                                                   \
      constexpr static bool NAME = false;                                      \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()

#define LLMX_DISPATCH_CUTE_DTYPE(TYPE, NAME, ...)                              \
  [&] {                                                                        \
    if (TYPE == ScalarType::kFP16) {                                           \
      using NAME = cute::half_t;                                               \
      return __VA_ARGS__();                                                    \
    } else if (TYPE == ScalarType::kBF16) {                                    \
      using NAME = cute::bfloat16_t;                                           \
      return __VA_ARGS__();                                                    \
    } else {                                                                   \
      assert(false);                                                           \
    }                                                                          \
  }()

#define LLMX_DISPATCH_HEAD_DIM(VALUE, NAME, ...)                               \
  [&] {                                                                        \
    if (VALUE <= 64) {                                                         \
      constexpr static int NAME = 64;                                          \
      return __VA_ARGS__();                                                    \
    } else if (VALUE <= 128) {                                                 \
      constexpr static int NAME = 128;                                         \
      return __VA_ARGS__();                                                    \
    } else if (VALUE <= 256) {                                                 \
      constexpr static int NAME = 256;                                         \
      return __VA_ARGS__();                                                    \
    } else {                                                                   \
      assert(false);                                                           \
    }                                                                          \
  }()
