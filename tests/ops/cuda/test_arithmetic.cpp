#include "tiny_llm/ops/cuda/arithmetic.hpp"
#include "tiny_llm/tensor/tensor.hpp"
#include "gtest/gtest.h"
#include <cmath>
#include <numeric>

namespace tiny_llm {
namespace {
void check_arithmetic_res(const Tensor &tensor, float target) {
  auto tmp = tensor.to({.type = DeviceType::kCpu, .id = 0});
  ThreadCudaContexts::Synchronize();
  const auto *ptr = tmp.data<float>();
  auto element_size =
      std::accumulate(tmp.shape().begin(), tmp.shape().end(), 1,
                      [](auto a, auto b) -> auto { return a * b; });

  for (int32_t i = 0; i < element_size; ++i) {
    EXPECT_FLOAT_EQ(ptr[i], target);
  }
}
} // namespace

TEST(CudaOps, Arithmetic) {
  std::vector<int64_t> shape{2, 3, 40, 50};
  size_t element_size = static_cast<size_t>(2) * 3 * 40 * 50;

  Tensor a({.type = DeviceType::kCpu, .id = 0}, DataType::kFloat32, shape);
  Tensor b({.type = DeviceType::kCpu, .id = 0}, DataType::kFloat32, shape);
  Tensor c({.type = DeviceType::kCpu, .id = 0}, DataType::kFloat32, shape);
  auto *ptr_a = a.data<float>();
  auto *ptr_b = b.data<float>();
  auto *ptr_c = c.data<float>();
  for (uint32_t i = 0; i < element_size; ++i) {
    ptr_a[i] = 1.F;
    ptr_b[i] = 10.F;
    ptr_c[i] = 100.F;
  }
  a = a.to({.type = DeviceType::kCuda, .id = 0});
  b = b.to({.type = DeviceType::kCuda, .id = 0});
  c = c.to({.type = DeviceType::kCuda, .id = 0});

  cuda::arithmetic(a.data<float>(), a.data<float>(), a.data<float>(),
                   element_size, cuda::ArithmeticType::kAdd);
  check_arithmetic_res(a, 2.F);
  cuda::arithmetic(a.data<float>(), c.data<float>(), c.data<float>(),
                   element_size, cuda::ArithmeticType::kMul);
  check_arithmetic_res(c, 200.F);
  cuda::arithmetic(b.data<float>(), a.data<float>(), b.data<float>(),
                   element_size, cuda::ArithmeticType::kSub);
  check_arithmetic_res(b, 8.F);
  cuda::arithmetic(c.data<float>(), b.data<float>(), a.data<float>(),
                   element_size, cuda::ArithmeticType::kDiv);
  check_arithmetic_res(a, 25.F);
}
} // namespace tiny_llm
