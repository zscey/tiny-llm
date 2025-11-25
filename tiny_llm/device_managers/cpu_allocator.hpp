#pragma once

#include "tiny_llm/common/common_macros.hpp"
#include "tiny_llm/device_managers/buffer.hpp"

namespace tiny_llm {
class CpuAllocator {
public:
  CpuAllocator() = default;
  TINY_LLM_DELETE_COPY_MOVE(CpuAllocator);
  ~CpuAllocator() = default;

  static auto Allocate(std::size_t size, std::size_t alignment = 32) -> Buffer;

  static void DefaultDeleter(void *ptr);
};
} // namespace tiny_llm
