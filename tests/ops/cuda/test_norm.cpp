#include "tiny_llm/ops/cuda/norm.hpp"
#include "tiny_llm/tensor/tensor.hpp"
#include "tiny_llm/utils/runfile.hpp"
#include "tiny_llm/weight_managers/safetensors/weight_manager.hpp"
#include "gtest/gtest.h"

namespace tiny_llm {
TEST(CudaOps, RMSNorm) {
  int64_t b = 2;
  int64_t l = 107;
  int64_t d = 241;

  Tensor input({.type = DeviceType::kCpu}, DataType::kFloat32, {b, l, d});
  {
    auto *input_ptr = input.data<float>();
    for (size_t i = 0, i_end = static_cast<size_t>(b * l * d); i < i_end; ++i) {
      input_ptr[i] = static_cast<float>(i % 100) / 100.F;
    }
    input = input.to({.type = DeviceType::kCuda, .id = 0});
  }

  Tensor weight({.type = DeviceType::kCpu}, DataType::kFloat32, {d});
  {
    auto *weight_ptr = weight.data<float>();
    for (size_t i = 0, i_end = static_cast<size_t>(d); i < i_end; ++i) {
      weight_ptr[i] = static_cast<float>(i % 100) / 100.F;
    }
    weight = weight.to({.type = DeviceType::kCuda, .id = 0});
  }

  Tensor dst({.type = DeviceType::kCuda, .id = 0}, DataType::kFloat32,
             {b, l, d});
  cuda::rms_norm(input.data<float>(), weight.data<float>(), dst.data<float>(),
                 b * l, d, 1.192e-7F);
  dst = dst.to({.type = DeviceType::kCpu, .id = 0});

  ThreadCudaContexts::Synchronize();

  SafeTensorWeightManager wm(utils::BazelRunfile::RLocation(
      "tiny_llm/tests/datas/rms_norm_b2l107d241.safetensors"));
  const auto *dst_ptr = dst.data<float>();
  const auto *target_ptr =
      reinterpret_cast<const float *>(wm.get_tensor("res").data);
  for (size_t i = 0, i_end = static_cast<size_t>(b * l * d); i < i_end; ++i) {
    EXPECT_TRUE(std::abs(dst_ptr[i] - target_ptr[i]) < 1e-5F);
  }
}
} // namespace tiny_llm
