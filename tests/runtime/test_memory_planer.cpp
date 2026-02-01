#include "tiny_llm/runtime/greedy_memory_planer.hpp"
#include "gtest/gtest.h"

namespace tiny_llm {
TEST(Runtime, GreedyMemoryPlaner) {
  GreedyMemoryPlaner mp;

  EXPECT_ANY_THROW(mp.deallocate({.offset = 0, .size = 10}));

  {
    auto v_block = mp.allocate(3, 256);
    EXPECT_TRUE(v_block.offset == 0);
    EXPECT_TRUE(v_block.size == 3);
    EXPECT_TRUE(mp.total_size() == 3);
    EXPECT_TRUE(mp.v_blocks().empty());
  }
  {
    auto v_block = mp.allocate(3, 256);
    EXPECT_TRUE(v_block.offset == 256);
    EXPECT_TRUE(v_block.size == 3);
    EXPECT_TRUE(mp.total_size() == 259);
    EXPECT_TRUE(mp.v_blocks().size() == 1);
    EXPECT_TRUE(mp.v_blocks().begin()->offset == 3);
    EXPECT_TRUE(mp.v_blocks().begin()->size == 253);
  }
  {
    auto v_block = mp.allocate(5, 128);
    EXPECT_TRUE(v_block.offset == 128);
    EXPECT_TRUE(v_block.size == 5);
    EXPECT_TRUE(mp.total_size() == 259);
    EXPECT_TRUE(mp.v_blocks().size() == 2);
    EXPECT_TRUE(mp.v_blocks().begin()->offset == 3);
    EXPECT_TRUE(mp.v_blocks().begin()->size == 125);
    EXPECT_TRUE(std::next(mp.v_blocks().begin())->offset == 133);
    EXPECT_TRUE(std::next(mp.v_blocks().begin())->size == 123);
  }

  EXPECT_NO_THROW(mp.deallocate({.offset = 0, .size = 3}));
  EXPECT_NO_THROW(mp.deallocate({.offset = 256, .size = 3}));
  EXPECT_NO_THROW(mp.deallocate({.offset = 128, .size = 5}));
}

} // namespace tiny_llm
