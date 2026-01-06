#pragma once

#include "tiny_llm/device_managers/buffer.hpp"

namespace tiny_llm {
class CudaAllocator {
public:
  /// @brief Allocate the Buffer on the current device.
  static auto Allocate(std::size_t size) -> Buffer;
};

class CudaHostAllocator {
public:
  static auto Allocate(std::size_t size) -> Buffer;
};
} // namespace tiny_llm
