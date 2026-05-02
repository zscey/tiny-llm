#pragma once

#include "tiny_llm/common/construct_macros.hpp"
#include <cstdint>
#include <memory>
#include <utility>

namespace tiny_llm {
enum class DeviceType : std::uint8_t {
  kCpu,
  kCudaHost,
  kCuda,
};

auto to_string(DeviceType dev_type) -> std::string;

using DeviceId = std::int8_t;

struct Device {
  DeviceType type{DeviceType::kCpu};
  // -1 indicates the current device, which is useful in scenarios such as CUDA.
  DeviceId id{-1};
};

using DeleterFnPtr = void (*)(void *);

class Buffer {
private:
  void *ptr_{};
  std::size_t size_{};
  Device device_{};
  DeleterFnPtr deleter_{};

public:
  Buffer() = default;

  Buffer(void *ptr, std::size_t size, Device device, DeleterFnPtr deleter = {});

  Buffer(const Buffer &) = delete;

  Buffer(Buffer &&other) noexcept;

  auto operator=(const Buffer &) -> Buffer & = delete;

  auto operator=(Buffer &&other) noexcept -> Buffer &;

  ~Buffer() noexcept;

  auto get_ptr() -> void * { return ptr_; }

  [[nodiscard]] auto get_ptr() const -> const void * { return ptr_; }

  [[nodiscard]] auto get_size() const -> std::size_t { return size_; }

  [[nodiscard]] auto get_device() const -> Device { return device_; }
};
} // namespace tiny_llm
