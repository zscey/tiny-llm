#include "tiny_llm/ops/cpu/cpu_op_common.hpp"
#include "gtest/gtest.h"
#include <numeric>

namespace tiny_llm {

TEST(CpuOps, Common) {
  std::vector<size_t> vec(1000);
  std::iota(vec.begin(), vec.end(), 0);
  auto res = cpu::parallel_run(
      [&vec](size_t beg, size_t end) -> auto {
        size_t res{};
        while (beg < end) {
          res += vec.at(beg);
          ++beg;
        }
        return res;
      },
      0, 1000, 10);

  EXPECT_TRUE(res.size() == 10);
  for (size_t i = 0; i < 10; ++i) {
    EXPECT_TRUE(res.at(i) == (i * 200 + 99) * 50);
  }

  res.resize(5, 0);
  cpu::parallel_run(
      [&res](size_t beg, size_t end) -> auto { res.at(beg) = beg + end; }, 0, 5,
      5);
  for (size_t i = 0; i < 5; ++i) {
    res.at(i) = (2 * i) + 1;
  }
}
} // namespace tiny_llm
