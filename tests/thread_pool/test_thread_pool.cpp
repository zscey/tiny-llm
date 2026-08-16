#include "tiny_llm/thread_pool/thread_pool.hpp"
#include "gtest/gtest.h"
#include <numeric>

namespace tiny_llm {
namespace {
auto sum(const std::vector<size_t> &vec, size_t beg, size_t end) -> size_t {
  size_t res{};
  while (beg < end) {
    res += vec.at(beg);
    ++beg;
  }
  return res;
}
} // namespace

TEST(ThreadPool, ThreadPool) {
  ThreadPool pool;
  std::vector<size_t> vec(1000);
  std::iota(vec.begin(), vec.end(), 0);

  std::vector<std::future<size_t>> futures;
  futures.reserve(10);
  for (size_t i = 0; i < 10; ++i) {
    futures.emplace_back(pool.enqueue(sum, vec, i * 100, (i + 1) * 100));
  }

  for (size_t i = 0; i < 10; ++i) {
    EXPECT_TRUE(futures.at(i).get() == ((i * 200) + 99) * 50);
  }
}
} // namespace tiny_llm
