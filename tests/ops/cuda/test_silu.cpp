#include "tiny_llm/ops/cuda/silu.hpp"
#include "tiny_llm/tensor/tensor.hpp"
#include "gtest/gtest.h"
#include <cmath>

namespace tiny_llm {
namespace {
auto silu(float f) -> float { return f / (1 + std::exp(-f)); }
} // namespace

TEST(CudaOps, SiLU) {
  std::vector<int64_t> shape{1, 2};
  Device device{.type = DeviceType::kCuda, .id = 0};

  Tensor src({.type = DeviceType::kCpu}, DataType::kFloat32, shape);
  auto *src_ptr = src.data<float>();
  src_ptr[0] = 0.F;
  src_ptr[1] = 1.F;
  src = src.to(device);
  Tensor dst(device, DataType::kFloat32, shape);

  cuda::silu(src.data<float>(), dst.data<float>(), 2);
  dst = dst.to({.type = DeviceType::kCpu});
  ThreadCudaContexts::SynchronizeDevice();

  const auto *dst_ptr = dst.data<float>();
  EXPECT_FLOAT_EQ(dst_ptr[0], silu(0.F));
  EXPECT_FLOAT_EQ(dst_ptr[1], silu(1.F));
}

} // namespace tiny_llm
