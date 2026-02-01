#pragma once

#include "tiny_llm/runtime/memory_planer.hpp"
#include <list>

namespace tiny_llm {
class GreedyMemoryPlaner {
public:
  auto allocate(size_t size, size_t alignment) -> VirtualBlock;
  void deallocate(VirtualBlock v_block);

  [[nodiscard]] auto v_blocks() const -> const std::list<VirtualBlock> & {
    return v_blocks_;
  }

  [[nodiscard]] auto total_size() const -> size_t { return total_size_; }

private:
  size_t total_size_{};
  std::list<VirtualBlock> v_blocks_;
};
} // namespace tiny_llm
