#include "tiny_llm/runtime/greedy_memory_planer.hpp"
#include "tiny_llm/common/checks.hpp"
#include "tiny_llm/common/exception.hpp"
#include <stdexcept>

namespace tiny_llm {
namespace {
auto aligned_pos(size_t pos, size_t alignment) -> size_t {
  return (pos + alignment - 1) / alignment * alignment;
}
} // namespace

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
auto GreedyMemoryPlaner::allocate(size_t size, size_t alignment)
    -> VirtualBlock {
  TINY_LLM_CHECK(tiny_llm::InvalidArgumentError, size > 0);
  TINY_LLM_CHECK(tiny_llm::InvalidArgumentError, alignment > 0);

  auto iter = v_blocks_.begin();
  while (iter != v_blocks_.end()) {
    auto offset = aligned_pos(iter->offset, alignment);
    if (offset + size <= iter->offset + iter->size) {
      if (offset + size < iter->offset + iter->size) {
        v_blocks_.emplace(std::next(iter), offset + size,
                          iter->offset + iter->size - offset - size);
      }
      iter->size = offset - iter->offset;
      if (iter->size == 0) {
        v_blocks_.erase(iter);
      }
      return {.offset = offset, .size = size};
    }
    std::advance(iter, 1);
  }

  auto offset = aligned_pos(total_size_, alignment);
  if (total_size_ < offset) {
    if (!v_blocks_.empty() &&
        v_blocks_.back().offset + v_blocks_.back().size == total_size_) {
      v_blocks_.back().size += offset - total_size_;
    } else {
      v_blocks_.emplace_back(total_size_, offset - total_size_);
    }
  }
  total_size_ = offset + size;
  return {.offset = offset, .size = size};
}

void GreedyMemoryPlaner::deallocate(VirtualBlock v_block) {
  TINY_LLM_CHECK(tiny_llm::RuntimeError,
                 v_block.offset + v_block.size <= total_size_);

  auto iter = v_blocks_.begin();
  while (iter != v_blocks_.end()) {
    if (v_block.offset + v_block.size <= iter->offset) {
      if (v_block.offset + v_block.size == iter->offset) {
        iter->offset = v_block.offset;
        iter->size += v_block.size;
      } else {
        iter = v_blocks_.emplace(iter, v_block);
      }

      if (iter != v_blocks_.begin()) {
        auto prev_iter = std::prev(iter);
        if (prev_iter->offset + prev_iter->size > iter->offset) {
          TINY_LLM_THROW_ERROR(tiny_llm::RuntimeError, "Bad virtual block.");
        }
        if (prev_iter->offset + prev_iter->size == iter->offset) {
          iter->offset = prev_iter->offset;
          iter->size += prev_iter->size;
          v_blocks_.erase(prev_iter);
        }
      }

      break;
    }
    std::advance(iter, 1);
  }

  if (iter == v_blocks_.end()) {
    if (!v_blocks_.empty() &&
        v_blocks_.back().offset + v_blocks_.back().size == v_block.offset) {
      v_blocks_.back().size += v_block.size;
    } else {
      if (!v_blocks_.empty()) {
        if (v_blocks_.back().offset + v_blocks_.back().size > v_block.offset) {
          TINY_LLM_THROW_ERROR(tiny_llm::RuntimeError, "Bad virtual block.");
        }
      }
      v_blocks_.emplace_back(v_block);
    }
  }
}

static_assert(MemoryPlanner<GreedyMemoryPlaner>);
} // namespace tiny_llm
