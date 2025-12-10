#pragma once

#include "tiny_llm/device_managers/buffer.hpp"

namespace tiny_llm {
class CudaAllocator {
public:
  static auto Allocate(std::size_t size) -> Buffer;
};

class CudaHostAllocator {
public:
  static auto Allocate(std::size_t size) -> Buffer;
};
} // namespace tiny_llm
