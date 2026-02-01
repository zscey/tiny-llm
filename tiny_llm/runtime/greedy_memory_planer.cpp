#include "tiny_llm/runtime/greedy_memory_planer.hpp"
#include "memory_planer.hpp"
#include "tiny_llm/common/log_and_excepts.hpp"
#include <stdexcept>

namespace tiny_llm {
namespace {
auto aligned_pos(size_t pos, size_t alignment) -> size_t {
  return (pos + alignment - 1) / alignment * alignment;
}

void merge(std::list<VirtualBlock> &v_blocks) {
  auto iter = v_blocks.begin();
  while (iter != v_blocks.end()) {
    auto next_iter = std::next(iter);
    if (next_iter == v_blocks.end()) {
      break;
    }

    if (iter->offset + iter->size == next_iter->offset) {
      iter->size += next_iter->size;
      v_blocks.erase(next_iter);
    } else {
      iter = next_iter;
    }
  }
}
} // namespace

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
auto GreedyMemoryPlaner::allocate(size_t size, size_t alignment)
    -> VirtualBlock {
  TINY_LLM_CHECK(alignment > 0);

  auto iter = v_blocks_.begin();
  while (iter != v_blocks_.end()) {
    auto offset = aligned_pos(iter->offset, alignment);
    if (offset + size <= iter->offset + iter->size) {
      if (offset + size < iter->offset + iter->size) {
        v_blocks_.emplace(std::next(iter), offset + size,
                          iter->offset + iter->size - offset - size);
      }
      if (iter->offset < offset) {
        iter->size = offset - iter->offset;
      }
      return {.offset = offset, .size = size};
    }
    std::advance(iter, 1);
  }

  auto offset = aligned_pos(total_size_, alignment);
  if (total_size_ < offset) {
    v_blocks_.emplace_back(total_size_, offset - total_size_);
    merge(v_blocks_);
  }
  total_size_ = offset + size;
  return {.offset = offset, .size = size};
}

void GreedyMemoryPlaner::deallocate(VirtualBlock v_block) {
  TINY_LLM_CHECK(v_block.offset + v_block.size <= total_size_);

  auto iter = v_blocks_.begin();
  while (iter != v_blocks_.end()) {
    if (v_block.offset + v_block.size <= iter->offset) {
      iter = v_blocks_.insert(iter, v_block);
      break;
    }
    std::advance(iter, 1);
  }
  if (iter == v_blocks_.end()) {
    v_blocks_.emplace_back(v_block);
    iter = std::prev(v_blocks_.end());
  }

  if (iter != v_blocks_.begin()) {
    auto prev_iter = std::prev(iter);
    if (prev_iter->offset + prev_iter->size > v_block.offset) {
      TINY_LLM_THROW_ERROR(std::runtime_error, "Bad virtual block.");
    }
  }

  merge(v_blocks_);
}

static_assert(MemoryPlanner<GreedyMemoryPlaner>);
} // namespace tiny_llm
