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
    return "Unknow";
  }
}

Buffer::Buffer(void *ptr, std::size_t size, Device device,
               std::unique_ptr<IDeleter> deleter)
    : ptr_(ptr), size_(size), device_(device), deleter_(std::move(deleter)) {}

Buffer::Buffer(Buffer &&other) noexcept {
  std::swap(ptr_, other.ptr_);
  std::swap(size_, other.size_);
  std::swap(device_, other.device_);
  std::swap(deleter_, other.deleter_);
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
  if (deleter_) {
    deleter_->cleanup(ptr_);
  }
}
} // namespace tiny_llm
