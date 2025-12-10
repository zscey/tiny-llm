#pragma once

#include "tiny_llm/device_managers/buffer.hpp"

namespace tiny_llm {
class CpuAllocator {
public:
  static auto Allocate(std::size_t size, std::size_t alignment = 32) -> Buffer;
};
} // namespace tiny_llm
