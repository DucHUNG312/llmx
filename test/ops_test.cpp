#include "llmx/ops.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

namespace llmx {

TEST(OpTest, ToFP32) {
  const std::vector<int> dims = {4};
  Tensor input(dims, ScalarType::kFP16, Device(DeviceType::kCUDA, 0));
  input.allocate();

  // Fill input with FP16 values [1.0, 2.0, 3.0, 4.0]
  auto *in_ptr = input.data().ptr<__half>();
  for (int i = 0; i < 4; ++i) {
    in_ptr[i] = __float2half(static_cast<float>(i + 1));
  }

  Tensor output;
  to_fp32(output, input);
  cudaDeviceSynchronize();

  EXPECT_TRUE(output.is_valid());
  EXPECT_EQ(output.dtype(), ScalarType::kFP32);
  EXPECT_EQ(output.dims(), dims);
  EXPECT_EQ(output.device(), Device(DeviceType::kCUDA, 0));

  const auto *out_ptr = output.data().ptr<float>();
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(out_ptr[i], static_cast<float>(i + 1));
  }
}

} // namespace llmx
