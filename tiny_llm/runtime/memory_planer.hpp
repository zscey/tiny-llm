#pragma once

#include "tiny_llm/common/common_macros.hpp"
#include <concepts>
#include <cstddef>

namespace tiny_llm {
struct VirtualBlock {
  size_t offset{};
  size_t size{};
};

template <typename T>
concept MemoryPlanner =
    requires(T t, size_t size, size_t alignment, VirtualBlock v_block) {
      { t.allocate(size, alignment) } -> std::same_as<VirtualBlock>;
      { t.deallocate(v_block) } -> std::same_as<void>;
    };
} // namespace tiny_llm
