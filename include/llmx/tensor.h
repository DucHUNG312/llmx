#pragma once

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <numeric>
#include <optional>
#include <unordered_map>
#include <vector>

#if defined(_MSC_VER)
#include <malloc.h>
#endif

namespace llmx {

enum class ScalarType { kFP32, kFP16, kBF16 };

size_t get_scalar_size(ScalarType dtype);

enum class DeviceType { kCPU, kCUDA };

struct Device {
  DeviceType type;
  int id;

  Device() = default;
  Device(DeviceType type, int id = -1) : type(type), id(id) {}

  bool operator==(const Device &other) const {
    return type == other.type && id == other.id;
  }
  bool operator!=(const Device &other) const { return !(*this == other); }
};

#define cpu0 Device(DeviceType::kCPU)
#define cuda0 Device(DeviceType::kCUDA, 0)
#define cuda1 Device(DeviceType::kCUDA, 1)
#define cuda2 Device(DeviceType::kCUDA, 2)
#define cuda3 Device(DeviceType::kCUDA, 3)
#define cuda4 Device(DeviceType::kCUDA, 4)
#define cuda5 Device(DeviceType::kCUDA, 5)
#define cuda6 Device(DeviceType::kCUDA, 6)
#define cuda7 Device(DeviceType::kCUDA, 7)

struct MemInfo {
  void *data;
  std::size_t nbytes;
  Device device;
};

class Allocator {
public:
  void *allocate(size_t nbytes, Device device);
  void free(void *data);
  std::optional<MemInfo> mem_info(void *data);

private:
  std::unordered_map<void *, MemInfo> mmap_;
};

static Allocator *get_llmx_allocator() {
  static Allocator allocator;
  return &allocator;
}

void *cpu_aligned_malloc(std::size_t size, std::size_t alignment = 64);
void cpu_aligned_free(void *ptr);
void *cuda_allocate_managed(size_t nbytes, Device device);
void cuda_free(void *ptr);

void *llmx_allocate(size_t nbytes, Device device);
void llmx_free(void *data);
std::optional<MemInfo> llmx_mem_info(void *data);
void llmx_copy(Device src_device, Device dst_device, const void *src, void *dst,
               size_t nbytes);
void llmx_copy(ScalarType src_dtype, ScalarType dst_dtype, const void *src,
               void *dst, size_t num);

class DataPtr {
public:
  DataPtr() = default;
  DataPtr(void *ptr);
  DataPtr(const DataPtr &other);
  DataPtr &operator=(const DataPtr &other);
  DataPtr(DataPtr &&other) noexcept;
  DataPtr &operator=(DataPtr &&other) noexcept;
  ~DataPtr();

  void *ptr() const noexcept { return ptr_; }
  void *ptr() noexcept { return ptr_; }
  template <typename T> const T *ptr() const noexcept {
    return reinterpret_cast<const T *>(ptr_);
  }
  template <typename T> T *ptr() noexcept {
    return reinterpret_cast<T *>(ptr_);
  }

  bool operator==(std::nullptr_t) noexcept {
    return ptr_ == nullptr && ref_count_ == nullptr;
  }
  bool operator!=(std::nullptr_t) noexcept {
    return !(this->operator==(std::nullptr_t{}));
  }

private:
  void increase_ref_count();
  void decrease_ref_count();

  void *ptr_ = nullptr;
  int *ref_count_ = nullptr;
};

class Tensor {
public:
  Tensor() = default;
  Tensor(const std::vector<int> &dims, ScalarType dtype, Device device);
  Tensor(void *data, const std::vector<int> &dims, ScalarType dtype,
         Device device);

  ScalarType dtype() const noexcept { return dtype_; }
  void dtype(ScalarType dtype) noexcept { dtype_ = dtype; }
  std::vector<int> dims() const noexcept { return dims_; }
  int dim(int axis) const noexcept { return dims_[axis]; }
  int stride(int axis) const noexcept { return strides_[axis]; }
  int ndim() const noexcept { return dims_.size(); }
  Device device() const noexcept { return device_; }
  const DataPtr &data() const noexcept { return data_; }
  DataPtr &data() noexcept { return data_; }
  const void *const_data_ptr() const noexcept { return data_.ptr(); }
  void *data_ptr() noexcept { return data_.ptr(); }
  template <typename T> const T *const_data_ptr() const noexcept {
    return data_.template ptr<T>();
  }
  template <typename T> T *data_ptr() noexcept {
    return data_.template ptr<T>();
  }

  bool is_valid() const noexcept {
    return data_.ptr() != nullptr && capacity_ > 0;
  }
  bool contiguous() const noexcept {
    if (strides_.empty())
      return true;
    if (strides_.back() != 1)
      return false;
    for (int i = static_cast<int>(dims_.size()) - 2; i >= 0; --i)
      if (strides_[i] != strides_[i + 1] * dims_[i + 1])
        return false;
    return true;
  }
  size_t nbytes() const noexcept { return capacity_ * get_scalar_size(dtype_); }
  size_t capacity() const noexcept { return capacity_; }
  size_t count(int axis = 0) const noexcept {
    return dims_[axis] * strides_[axis];
  }
  size_t numel() const noexcept {
    return std::accumulate(dims_.begin(), dims_.end(), size_t(1),
                           std::multiplies<>{});
  }

  void allocate();
  void resize(const std::vector<int> &dims);
  void to_device(Device device, bool transfer = true);

private:
  DataPtr data_;
  ScalarType dtype_;
  std::vector<int> dims_;
  std::vector<int> strides_;
  size_t capacity_;
  Device device_ = Device(DeviceType::kCPU);
};

} // namespace llmx