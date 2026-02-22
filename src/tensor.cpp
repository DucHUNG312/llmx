#include "llmx/tensor.h"
#include "llmx/dispatch.h"
#include "llmx/scalar.h"
#include "llmx/utility.h"
#include <cstddef>
#include <cuda_runtime.h>

namespace llmx {

size_t get_scalar_size(ScalarType dtype) {
  switch (dtype) {
  case ScalarType::kFP32:
    return sizeof(float);
  case ScalarType::kFP16:
    return sizeof(__half);
  case ScalarType::kBF16:
    return sizeof(__nv_bfloat16);
  default:
    LLMX_CHECK(false, "Unsupported data type");
    return -1;
  }
}

void *cpu_aligned_malloc(std::size_t size, std::size_t alignment) {
#if defined(_MSC_VER)
  void *ptr = _aligned_malloc(size, alignment);
  if (!ptr) {
    throw std::bad_alloc();
  }
  std::memset(ptr, 0, size);
  return ptr;
#elif defined(__APPLE__) || defined(__linux__)
  void *ptr = nullptr;
  if (alignment < sizeof(void *)) {
    alignment = sizeof(void *);
  }
  int res = posix_memalign(&ptr, alignment, size);
  if (res != 0) {
    throw std::bad_alloc();
  }
  std::memset(ptr, 0, size);
  return ptr;
#else
  void *ptr = aligned_alloc(alignment, size);
  if (!ptr) {
    throw std::bad_alloc();
  }
  std::memset(ptr, 0, size);
  return ptr;
#endif
}

inline void cpu_aligned_free(void *ptr) {
  if (!ptr) {
    return;
  }
#if defined(_MSC_VER)
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

inline void *cuda_allocate_managed(size_t nbytes, Device device) {
  void *ptr = nullptr;
  LLMX_CUDA_CHECK(cudaSetDevice(device.id));
  LLMX_CUDA_CHECK(cudaMallocManaged(&ptr, nbytes));
  LLMX_CUDA_CHECK(cudaMemset(ptr, 0, nbytes));
  return ptr;
}

inline void cuda_free(void *ptr) { LLMX_CUDA_CHECK(cudaFree(ptr)); }

void *llmx_allocate(size_t nbytes, Device device) {
  return get_llmx_allocator()->allocate(nbytes, device);
}

void llmx_free(void *data) { get_llmx_allocator()->free(data); }

std::optional<MemInfo> llmx_mem_info(void *data) {
  return get_llmx_allocator()->mem_info(data);
}

void llmx_copy(Device src_device, Device dst_device, const void *src, void *dst,
               size_t nbytes) {
  LLMX_CHECK(src != nullptr && dst != nullptr);
  if (src_device.type == DeviceType::kCPU &&
      dst_device.type == DeviceType::kCPU) {
    std::memcpy(dst, src, nbytes);
  } else {
    cudaMemcpyKind kind;
    if (src_device.type == DeviceType::kCPU &&
        dst_device.type == DeviceType::kCUDA) {
      kind = cudaMemcpyHostToDevice;
    } else if (src_device.type == DeviceType::kCUDA &&
               dst_device.type == DeviceType::kCPU) {
      kind = cudaMemcpyDeviceToHost;
    } else {
      kind = cudaMemcpyDeviceToDevice;
    }
    const int cuda_device_id =
        (kind == cudaMemcpyDeviceToHost) ? src_device.id : dst_device.id;
    LLMX_CUDA_CHECK(cudaSetDevice(cuda_device_id));
    LLMX_CUDA_CHECK(cudaMemcpy(dst, src, nbytes, kind));
  }
}

void llmx_copy(ScalarType src_dtype, ScalarType dst_dtype, const void *src,
               void *dst, size_t num) {
  LLMX_CHECK(src != nullptr && dst != nullptr);
  LLMX_DISPATCH_SCALAR_TYPE(src_dtype, SrcType, [&] {
    LLMX_DISPATCH_SCALAR_TYPE(dst_dtype, DsrType, [&] {
      ScalarConverter<SrcType, DsrType>{}(
          reinterpret_cast<const SrcType *>(src),
          reinterpret_cast<DsrType *>(dst), num);
    });
  });
}

void *Allocator::allocate(size_t nbytes, Device device) {
  void *ptr = nullptr;
  switch (device.type) {
  case DeviceType::kCPU:
    ptr = cpu_aligned_malloc(nbytes);
    break;
  case DeviceType::kCUDA:
    ptr = cuda_allocate_managed(nbytes, device);
    break;
  }
  mmap_[ptr] = MemInfo{ptr, nbytes, device};
  return ptr;
}

void Allocator::free(void *data) {
  if (const auto it = mmap_.find(data); it != mmap_.end()) {
    switch (it->second.device.type) {
    case DeviceType::kCPU:
      cpu_aligned_free(data);
      break;
    case DeviceType::kCUDA:
      cuda_free(data);
      break;
    }
    mmap_.erase(it);
  }
}

std::optional<MemInfo> Allocator::mem_info(void *data) {
  if (auto it = mmap_.find(data); it != mmap_.end()) {
    return it->second;
  }
  return std::nullopt;
}

DataPtr::DataPtr(void *ptr) : ptr_(ptr) {
  // if this is not external data, create ref_count to manage it
  if (llmx_mem_info(ptr).has_value()) {
    ref_count_ = new int(1);
  }
}

DataPtr::DataPtr(const DataPtr &other)
    : ptr_(other.ptr_), ref_count_(other.ref_count_) {
  increase_ref_count();
}

DataPtr &DataPtr::operator=(const DataPtr &other) {
  if (this != &other) {
    if (ptr_ != nullptr) {
      decrease_ref_count();
    }
    ptr_ = other.ptr_;
    ref_count_ = other.ref_count_;
    increase_ref_count();
  }
  return *this;
}

DataPtr::DataPtr(DataPtr &&other) noexcept
    : ptr_(other.ptr_), ref_count_(other.ref_count_) {
  other.ptr_ = nullptr;
  other.ref_count_ = nullptr;
}

DataPtr &DataPtr::operator=(DataPtr &&other) noexcept {
  if (this != &other) {
    if (ptr_ != nullptr) {
      decrease_ref_count();
    }
    ptr_ = other.ptr_;
    ref_count_ = other.ref_count_;
    other.ptr_ = nullptr;
    other.ref_count_ = nullptr;
  }
  return *this;
}

DataPtr::~DataPtr() { decrease_ref_count(); }

void DataPtr::increase_ref_count() {
  if (ref_count_ != nullptr) {
    (*ref_count_)++;
  }
}
void DataPtr::decrease_ref_count() {
  if (ref_count_ != nullptr && (*ref_count_)-- == 1) {
    llmx_free(ptr_);
    delete ref_count_;
    ref_count_ = nullptr;
  }
}

Tensor::Tensor(const std::vector<int> &dims, ScalarType dtype, Device device)
    : dtype_(dtype), device_(device) {
  resize(dims);
}

Tensor::Tensor(void *data, const std::vector<int> &dims, ScalarType dtype,
               Device device)
    : dtype_(dtype), device_(device) {
  resize(dims);
  data_ = DataPtr(data);
  capacity_ = count();
}

void Tensor::allocate() {
  if (!is_valid()) {
    const auto nbytes = count() * get_scalar_size(dtype_);
    void *new_ptr = llmx_allocate(nbytes, device_);
    data_ = DataPtr(new_ptr);
    capacity_ = count();
  } else {
    const size_t new_nbytes = count() * get_scalar_size(dtype_);
    const size_t old_nbytes = nbytes();
    if (new_nbytes > old_nbytes) {
      void *new_ptr = llmx_allocate(new_nbytes, device_);
      data_ = DataPtr(new_ptr);
      capacity_ = count();
    }
  }
}

void Tensor::resize(const std::vector<int> &dims) {
  LLMX_CHECK(!dims.empty(), "dims must not be empty");
  dims_ = dims;
  strides_.resize(dims_.size());
  strides_.back() = 1;
  for (int i = dims_.size() - 2; i >= 0; i--) {
    strides_[i] = dims_[i + 1] * strides_[i + 1];
  }
}

void Tensor::to_device(Device device, bool transfer) {
  if (device_ != device && transfer) {
    const auto nbytes = count() * get_scalar_size(dtype_);
    auto *ptr = llmx_allocate(nbytes, device);
    llmx_copy(device_, device, data_.ptr(), ptr, nbytes);
    data_ = DataPtr(ptr);
  }
  device_ = device;
}

} // namespace llmx