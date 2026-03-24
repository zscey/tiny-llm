#include "tiny_llm/runtime/greedy_memory_planer.hpp"
#include "gtest/gtest.h"

namespace tiny_llm {
TEST(Runtime, GreedyMemoryPlaner) {
  GreedyMemoryPlaner mp;

  EXPECT_ANY_THROW(mp.deallocate({.offset = 0, .size = 10}));

  VirtualBlock v_block_1;
  {
    v_block_1 = mp.allocate(3, 256);
    EXPECT_TRUE(v_block_1.offset == 0);
    EXPECT_TRUE(v_block_1.size == 3);
    EXPECT_TRUE(mp.total_size() == 3);
    EXPECT_TRUE(mp.v_blocks().empty());
  }
  {
    auto v_block_2 = mp.allocate(3, 256);
    EXPECT_TRUE(v_block_2.offset == 256);
    EXPECT_TRUE(v_block_2.size == 3);
    EXPECT_TRUE(mp.total_size() == 259);
    EXPECT_TRUE(mp.v_blocks().size() == 1);
    EXPECT_TRUE(mp.v_blocks().begin()->offset == 3);
    EXPECT_TRUE(mp.v_blocks().begin()->size == 253);

    EXPECT_NO_THROW(mp.deallocate(v_block_2));
    EXPECT_TRUE(mp.total_size() == 259);
    EXPECT_TRUE(mp.v_blocks().size() == 1);
    EXPECT_TRUE(mp.v_blocks().begin()->offset == 3);
    EXPECT_TRUE(mp.v_blocks().begin()->size == 256);
  }
  VirtualBlock v_block_3;
  {
    v_block_3 = mp.allocate(8, 512);
    EXPECT_TRUE(v_block_3.offset == 512);
    EXPECT_TRUE(v_block_3.size == 8);
    EXPECT_TRUE(mp.total_size() == 520);
    EXPECT_TRUE(mp.v_blocks().size() == 1);
    EXPECT_TRUE(mp.v_blocks().begin()->offset == 3);
    EXPECT_TRUE(mp.v_blocks().begin()->size == 509);
  }
  VirtualBlock v_block_4;
  {
    v_block_4 = mp.allocate(10, 3);
    EXPECT_TRUE(v_block_4.offset == 3);
    EXPECT_TRUE(v_block_4.size == 10);
    EXPECT_TRUE(mp.total_size() == 520);
    EXPECT_TRUE(mp.v_blocks().size() == 1);
    EXPECT_TRUE(mp.v_blocks().begin()->offset == 13);
    EXPECT_TRUE(mp.v_blocks().begin()->size == 499);
  }
  VirtualBlock v_block_5;
  {
    v_block_5 = mp.allocate(12, 500);
    EXPECT_TRUE(v_block_5.offset == 500);
    EXPECT_TRUE(v_block_5.size == 12);
    EXPECT_TRUE(mp.total_size() == 520);
    EXPECT_TRUE(mp.v_blocks().size() == 1);
    EXPECT_TRUE(mp.v_blocks().begin()->offset == 13);
    EXPECT_TRUE(mp.v_blocks().begin()->size == 487);
  }

  {
    mp.deallocate(v_block_1);
    EXPECT_TRUE(mp.total_size() == 520);
    EXPECT_TRUE(mp.v_blocks().size() == 2);
    EXPECT_TRUE(mp.v_blocks().begin()->offset == 0);
    EXPECT_TRUE(mp.v_blocks().begin()->size == 3);
    EXPECT_TRUE(std::next(mp.v_blocks().begin())->offset == 13);
    EXPECT_TRUE(std::next(mp.v_blocks().begin())->size == 487);
  }
  {
    mp.deallocate(v_block_3);
    EXPECT_TRUE(mp.total_size() == 520);
    EXPECT_TRUE(mp.v_blocks().size() == 3);
    EXPECT_TRUE(mp.v_blocks().begin()->offset == 0);
    EXPECT_TRUE(mp.v_blocks().begin()->size == 3);
    EXPECT_TRUE(std::next(mp.v_blocks().begin())->offset == 13);
    EXPECT_TRUE(std::next(mp.v_blocks().begin())->size == 487);
    EXPECT_TRUE(std::next(std::next(mp.v_blocks().begin()))->offset == 512);
    EXPECT_TRUE(std::next(std::next(mp.v_blocks().begin()))->size == 8);
  }
  {
    mp.deallocate(v_block_5);
    EXPECT_TRUE(mp.total_size() == 520);
    EXPECT_TRUE(mp.v_blocks().size() == 2);
    EXPECT_TRUE(mp.v_blocks().begin()->offset == 0);
    EXPECT_TRUE(mp.v_blocks().begin()->size == 3);
    EXPECT_TRUE(std::next(mp.v_blocks().begin())->offset == 13);
    EXPECT_TRUE(std::next(mp.v_blocks().begin())->size == 507);
  }
  {
    mp.deallocate(v_block_4);
    EXPECT_TRUE(mp.total_size() == 520);
    EXPECT_TRUE(mp.v_blocks().size() == 1);
    EXPECT_TRUE(mp.v_blocks().begin()->offset == 0);
    EXPECT_TRUE(mp.v_blocks().begin()->size == 520);
  }

  {
    // EXPECT_ANY_THROW(mp.deallocate({.offset = 1000, .size = 10}));
    mp.allocate(20, 50);
    mp.allocate(30, 30);
    EXPECT_ANY_THROW(mp.deallocate({.offset = 30, .size = 31}));
    EXPECT_ANY_THROW(mp.deallocate({.offset = 29, .size = 30}));
  }
}

} // namespace tiny_llm
