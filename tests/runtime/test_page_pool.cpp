#include "tiny_llm/runtime/page_pool.hpp"
#include "gtest/gtest.h"

namespace tiny_llm {
TEST(Runtime, PagePool) {
  EXPECT_ANY_THROW(PagePool(PagePoolConfig{}));

  auto page_pool = PagePool(PagePoolConfig{
      .num_pages = 10, .num_heads = 4, .page_size = 2, .head_dim = 64});

  EXPECT_TRUE(page_pool.k_pool_ptr() != nullptr);
  EXPECT_TRUE(page_pool.v_pool_ptr() != nullptr);
  EXPECT_TRUE(page_pool.free_count() == 10);
  EXPECT_TRUE(page_pool.total_pages() == 10);
  EXPECT_ANY_THROW(page_pool.allocate_pages(12));
  EXPECT_ANY_THROW(page_pool.free_pages(std::vector<int32_t>{0}));

  auto pages = page_pool.allocate_pages(10);
  EXPECT_TRUE(pages == (std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
  EXPECT_TRUE(page_pool.free_count() == 0);
  EXPECT_TRUE(page_pool.total_pages() == 10);
  page_pool.free_pages(pages);
}
} // namespace tiny_llm
