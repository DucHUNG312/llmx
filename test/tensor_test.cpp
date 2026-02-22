#include "llmx/tensor.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

namespace llmx {

static void fill_sequential(Tensor &t) {
  auto *ptr = t.data().ptr<float>();
  for (size_t i = 0; i < t.numel(); ++i) {
    ptr[i] = static_cast<float>(i + 1);
  }
}

TEST(TensorTest, ConstructCPU) {
  const std::vector<int> dims = {2, 3};
  Tensor t(dims, ScalarType::kFP32, Device(DeviceType::kCPU));

  EXPECT_EQ(t.dims(), dims);
  EXPECT_EQ(t.dtype(), ScalarType::kFP32);
  EXPECT_EQ(t.device(), Device(DeviceType::kCPU));
  EXPECT_EQ(t.numel(), 6u);
  EXPECT_FALSE(t.is_valid()); // resize only — not yet allocated

  t.allocate();

  EXPECT_TRUE(t.is_valid());
  EXPECT_NE(t.data().ptr(), nullptr);
  EXPECT_EQ(t.capacity(), 6u);
  EXPECT_EQ(t.nbytes(), 6 * sizeof(float));

  // zero-initialized
  const auto *ptr = t.data().ptr<float>();
  for (size_t i = 0; i < t.numel(); ++i) {
    EXPECT_EQ(ptr[i], 0.0f);
  }
}

TEST(TensorTest, ConstructCUDA) {
  const std::vector<int> dims = {2, 3};
  Tensor t(dims, ScalarType::kFP32, Device(DeviceType::kCUDA, 0));

  EXPECT_EQ(t.dims(), dims);
  EXPECT_EQ(t.dtype(), ScalarType::kFP32);
  EXPECT_EQ(t.device(), Device(DeviceType::kCUDA, 0));
  EXPECT_EQ(t.numel(), 6u);
  EXPECT_FALSE(t.is_valid());

  t.allocate();

  EXPECT_TRUE(t.is_valid());
  EXPECT_NE(t.data().ptr(), nullptr);
  EXPECT_EQ(t.nbytes(), 6 * sizeof(float));

  const auto *ptr = t.data().ptr<float>();
  for (size_t i = 0; i < t.numel(); ++i) {
    EXPECT_EQ(ptr[i], 0.0f);
  }
}

TEST(TensorTest, Resize_MetadataOnly) {
  const std::vector<int> dims = {4};
  Tensor t(dims, ScalarType::kFP32, Device(DeviceType::kCPU));

  // resize sets dims/strides but does not allocate
  EXPECT_EQ(t.dims(), dims);
  EXPECT_EQ(t.numel(), 4u);
  EXPECT_FALSE(t.is_valid());
  EXPECT_EQ(t.data().ptr(), nullptr);
}

TEST(TensorTest, Allocate_AllocatesMemory) {
  const std::vector<int> dims = {4};
  Tensor t(dims, ScalarType::kFP32, Device(DeviceType::kCPU));

  EXPECT_FALSE(t.is_valid());
  t.allocate();

  EXPECT_TRUE(t.is_valid());
  EXPECT_NE(t.data().ptr(), nullptr);
  EXPECT_EQ(t.capacity(), 4u);
}

TEST(TensorTest, Allocate_NoReallocIfSmaller) {
  // Shrinking: same buffer, capacity unchanged
  const std::vector<int> dims = {8};
  Tensor t(dims, ScalarType::kFP32, Device(DeviceType::kCPU));
  t.allocate();
  const void *ptr_before = t.data().ptr();
  const size_t cap_before = t.capacity(); // 8

  t.resize({4});
  t.allocate(); // count() < capacity, so no realloc

  EXPECT_EQ(t.dims(), (std::vector<int>{4}));
  EXPECT_EQ(t.numel(), 4u);
  EXPECT_EQ(t.data().ptr(), ptr_before); // same buffer
  EXPECT_EQ(t.capacity(), cap_before);   // capacity unchanged at 8
}

TEST(TensorTest, Allocate_ReallocIfLarger) {
  // Growing: new buffer, capacity updated
  const std::vector<int> dims = {4};
  Tensor t(dims, ScalarType::kFP32, Device(DeviceType::kCPU));
  t.allocate();
  const void *ptr_before = t.data().ptr();

  t.resize({8});
  t.allocate(); // count() > capacity, must reallocate

  EXPECT_NE(t.data().ptr(), ptr_before);
  EXPECT_EQ(t.capacity(), 8u);
}

TEST(TensorTest, Count_DefaultAxis) {
  const std::vector<int> dims = {2, 3, 4};
  Tensor t(dims, ScalarType::kFP32, Device(DeviceType::kCPU));

  // count(0) = dims[0] * strides[0] = 2 * 12 = 24 == numel()
  EXPECT_EQ(t.count(), t.numel());
  EXPECT_EQ(t.count(), 24u);
  // count(1) = dims[1] * strides[1] = 3 * 4 = 12
  EXPECT_EQ(t.count(1), 12u);
  // count(2) = dims[2] * strides[2] = 4 * 1 = 4
  EXPECT_EQ(t.count(2), 4u);
}

TEST(TensorTest, ToDevice_NewAllocation) {
  const std::vector<int> dims = {4};
  Tensor t(dims, ScalarType::kFP32, Device(DeviceType::kCPU));
  t.allocate();
  fill_sequential(t);

  t.to_device(Device(DeviceType::kCUDA, 0));

  EXPECT_EQ(t.device(), Device(DeviceType::kCUDA, 0));
  EXPECT_NE(t.data().ptr(), nullptr);
  const auto *ptr = t.data().ptr<float>();
  for (size_t i = 0; i < t.numel(); ++i) {
    EXPECT_EQ(ptr[i], static_cast<float>(i + 1));
  }
}

TEST(TensorTest, ToDevice_SameDevice) {
  const std::vector<int> dims = {4};
  Tensor t(dims, ScalarType::kFP32, Device(DeviceType::kCPU));
  t.allocate();
  fill_sequential(t);

  const void *ptr_before = t.data().ptr();
  t.to_device(Device(DeviceType::kCPU));

  EXPECT_EQ(t.data().ptr(), ptr_before);
  EXPECT_EQ(t.device(), Device(DeviceType::kCPU));
}

TEST(TensorTest, ToDevice_NoTransfer) {
  // transfer=false: device label updated, buffer unchanged
  const std::vector<int> dims = {4};
  Tensor t(dims, ScalarType::kFP32, Device(DeviceType::kCPU));
  t.allocate();
  fill_sequential(t);
  const void *old_ptr = t.data().ptr();

  t.to_device(Device(DeviceType::kCUDA, 0), /*transfer=*/false);

  EXPECT_EQ(t.device(), Device(DeviceType::kCUDA, 0));
  EXPECT_EQ(t.data().ptr(), old_ptr); // same buffer, no copy
}

TEST(TensorTest, ToDevice_RoundTrip) {
  const std::vector<int> dims = {4};
  Tensor t(dims, ScalarType::kFP32, Device(DeviceType::kCPU));
  t.allocate();
  fill_sequential(t);

  t.to_device(Device(DeviceType::kCUDA, 0));
  t.to_device(Device(DeviceType::kCPU));

  EXPECT_EQ(t.device(), Device(DeviceType::kCPU));
  const auto *ptr = t.data().ptr<float>();
  for (size_t i = 0; i < t.numel(); ++i) {
    EXPECT_EQ(ptr[i], static_cast<float>(i + 1));
  }
}

TEST(TensorTest, SharedOwnership) {
  const std::vector<int> dims = {4};
  Tensor t1(dims, ScalarType::kFP32, Device(DeviceType::kCPU));
  t1.allocate();
  fill_sequential(t1);

  Tensor t2 = t1;
  EXPECT_EQ(t1.data().ptr(), t2.data().ptr());
  // write through t1, visible through t2
  t1.data().ptr<float>()[0] = 99.0f;
  EXPECT_EQ(t2.data().ptr<float>()[0], 99.0f);
}

} // namespace llmx
