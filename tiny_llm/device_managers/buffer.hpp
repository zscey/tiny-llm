#pragma once

#include <cstdint>
#include <utility>

namespace tiny_llm {
enum class DeviceType : std::uint8_t {
  kCpu,
  kCuda,
};

using DeviceId = std::int8_t;

struct Device {
  DeviceType type{DeviceType::kCpu};
  DeviceId id{-1};
};

using DeleterPtr = void (*)(void *);

class Buffer {
private:
  void *ptr_{};
  std::size_t size_{};
  Device device_{};
  DeleterPtr deleter_{};

public:
  Buffer() = default;
  Buffer(void *ptr, std::size_t size, Device device, DeleterPtr deleter = {});
  Buffer(const Buffer &) = delete;
  Buffer(Buffer &&other) noexcept;
  auto operator=(const Buffer &) -> Buffer & = delete;
  auto operator=(Buffer &&other) noexcept -> Buffer &;
  ~Buffer() noexcept;

  auto get_ptr() -> void * { return ptr_; }
  [[nodiscard]] auto get_ptr() const -> const void * { return ptr_; }
};
} // namespace tiny_llm
