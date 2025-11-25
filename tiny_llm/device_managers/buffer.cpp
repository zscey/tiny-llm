#include "tiny_llm/device_managers/buffer.hpp"

namespace tiny_llm {
Buffer::Buffer(void *ptr, std::size_t size, Device device, DeleterPtr deleter)
    : ptr_(ptr), size_(size), device_(device), deleter_(deleter) {}

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
  if (deleter_ != nullptr) {
    deleter_(ptr_);
  }
}
} // namespace tiny_llm
