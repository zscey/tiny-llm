#pragma once

#include "tiny_llm/common/common_macros.hpp"
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
  DeviceId id{-1};
};

class IDeleter {
public:
  IDeleter() = default;

  TINY_LLM_DELETE_COPY_MOVE(IDeleter);

  virtual void cleanup(void *ptr) = 0;

  virtual ~IDeleter() = default;
};

class Buffer {
private:
  void *ptr_{};
  std::size_t size_{};
  Device device_{};
  std::unique_ptr<IDeleter> deleter_;

public:
  Buffer() = default;

  Buffer(void *ptr, std::size_t size, Device device,
         std::unique_ptr<IDeleter> deleter = {});

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
