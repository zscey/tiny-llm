#include "tiny_llm/ops/cpu/arithmetic.hpp"
#include "tiny_llm/tensor/tensor.hpp"
#include "gtest/gtest.h"
#include <cmath>
#include <numeric>

namespace tiny_llm {
namespace {
void check_arithmetic_res(const Tensor &tensor, float target) {
  const auto *ptr = tensor.data<float>();
  auto element_size =
      std::accumulate(tensor.shape().begin(), tensor.shape().end(), 1,
                      [](auto a, auto b) -> auto { return a * b; });

  for (int32_t i = 0; i < element_size; ++i) {
    EXPECT_FLOAT_EQ(ptr[i], target);
  }
}
} // namespace

TEST(CpuOps, Arithmetic) {
  std::vector<int64_t> shape{2, 3, 40, 50};
  size_t element_size = static_cast<size_t>(2) * 3 * 40 * 50;

  Tensor a({.type = DeviceType::kCpu, .id = 0}, DataType::kFloat32, shape);
  Tensor b({.type = DeviceType::kCpu, .id = 0}, DataType::kFloat32, shape);
  auto *ptr_a = a.data<float>();
  auto *ptr_b = b.data<float>();
  for (uint32_t i = 0; i < element_size; ++i) {
    ptr_a[i] = 1.F;
    ptr_b[i] = 10.F;
  }

  cpu::arithmetic(a.data<float>(), 1.F, a.data<float>(), element_size,
                  ArithmeticType::kAdd);
  check_arithmetic_res(a, 2.F);
  cpu::arithmetic(a.data<float>(), 100.F, b.data<float>(), element_size,
                  ArithmeticType::kMul);
  check_arithmetic_res(b, 200.F);
  cpu::arithmetic(b.data<float>(), 10.F, b.data<float>(), element_size,
                  ArithmeticType::kSub);
  check_arithmetic_res(b, 190.F);
  cpu::arithmetic(b.data<float>(), 19.F, a.data<float>(), element_size,
                  ArithmeticType::kDiv);
  check_arithmetic_res(a, 10.F);
}
} // namespace tiny_llm
