#include "tiny_llm/device_managers/buffer.hpp"

namespace tiny_llm {
auto to_string(DeviceType dev_type) -> std::string {
  switch (dev_type) {
  case DeviceType::kCpu:
    return "Cpu";
  case DeviceType::kCudaHost:
    return "CudaHost";
  case tiny_llm::DeviceType::kCuda:
    return "Cuda";
  default:
    return "Unknown";
  }
}

Buffer::Buffer(void *ptr, std::size_t size, Device device, DeleterFnPtr deleter)
    : ptr_(ptr), size_(size), device_(device), deleter_(deleter) {}

Buffer::Buffer(Buffer &&other) noexcept
    : ptr_(other.ptr_), size_(other.size_), device_(other.device_),
      deleter_(other.deleter_) {
  other.ptr_ = {};
  other.size_ = {};
  other.device_ = {};
  other.deleter_ = {};
}

auto Buffer::operator=(Buffer &&other) noexcept -> Buffer & {
  if (this != std::addressof(other)) {
    std::swap(ptr_, other.ptr_);
    std::swap(size_, other.size_);
    std::swap(device_, other.device_);
    std::swap(deleter_, other.deleter_);
  }
  return *this;
}

Buffer::~Buffer() noexcept {
  if (ptr_ != nullptr && deleter_ != nullptr) {
    deleter_(ptr_);
  }
}
} // namespace tiny_llm
