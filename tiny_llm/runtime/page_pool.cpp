#include "tiny_llm/runtime/page_pool.hpp"
#include "tiny_llm/common/checks.hpp"
#include "tiny_llm/common/exception.hpp"

namespace tiny_llm {
PagePool::PagePool(PagePoolConfig config) {
  TINY_LLM_CHECK(InvalidArgumentError, config.num_pages > 0);
  TINY_LLM_CHECK(InvalidArgumentError, config.num_heads > 0);
  TINY_LLM_CHECK(InvalidArgumentError, config.page_size > 0);
  TINY_LLM_CHECK(InvalidArgumentError, config.head_dim > 0);
  std::vector<int64_t> tensor_shape{static_cast<int64_t>(config.num_pages),
                                    static_cast<int64_t>(config.num_heads),
                                    static_cast<int64_t>(config.page_size),
                                    static_cast<int64_t>(config.head_dim)};

  k_buffer_ = Tensor({.type = config.device_type}, config.data_type,
                     tensor_shape, true);
  v_buffer_ = Tensor({.type = config.device_type}, config.data_type,
                     tensor_shape, true);

  auto page_id = static_cast<int32_t>(config.num_pages) - 1;
  free_list_.resize(config.num_pages);
  std::ranges::generate(free_list_,
                        [&page_id]() -> int32_t { return page_id--; });
}

auto PagePool::k_pool_ptr() -> void * { return k_buffer_.data(); }

auto PagePool::v_pool_ptr() -> void * { return v_buffer_.data(); }

auto PagePool::allocate_pages(uint32_t num_logical_pages)
    -> std::vector<int32_t> {
  TINY_LLM_CHECK(InvalidArgumentError, num_logical_pages <= free_list_.size());

  std::vector<int32_t> phys_pages;
  phys_pages.reserve(num_logical_pages);
  for (uint32_t i = 0; i < num_logical_pages; ++i) {
    phys_pages.emplace_back(free_list_.back());
    free_list_.pop_back();
  }

  return phys_pages;
}

void PagePool::free_pages(std::span<const int32_t> page_ids) {
  TINY_LLM_CHECK(RuntimeError,
                 std::ranges::all_of(page_ids, [this](int32_t id) -> bool {
                   return std::ranges::find(free_list_, id) == free_list_.end();
                 }));
  for (auto id : page_ids) {
    free_list_.emplace_back(id);
  }
}

auto PagePool::free_count() const -> uint32_t {
  return static_cast<uint32_t>(free_list_.size());
}

auto PagePool::total_pages() const -> uint32_t {
  return static_cast<uint32_t>(k_buffer_.shape()[0]);
}

} // namespace tiny_llm
