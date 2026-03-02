#include "tiny_llm/ops/cuda/rope.hpp"
#include "tiny_llm/tensor/tensor.hpp"
#include "tiny_llm/utils/runfile.hpp"
#include "tiny_llm/weight_managers/safetensors/weight_manager.hpp"
#include "gtest/gtest.h"

namespace tiny_llm {
TEST(CudaOps, Rope) {
  Tensor cos({.type = DeviceType::kCuda, .id = 0}, DataType::kFloat32, {2, 96},
             true);
  Tensor sin({.type = DeviceType::kCuda, .id = 0}, DataType::kFloat32, {2, 96},
             true);
  cuda::rope(cos.data<float>(), sin.data<float>(), 2, 96, 10000);
  cos = cos.to({.type = DeviceType::kCpu, .id = 0});
  sin = sin.to({.type = DeviceType::kCpu, .id = 0});

  ThreadCudaContexts::Synchronize();

  SafeTensorWeightManager wm(utils::BazelRunfile::RLocation(
      "tiny_llm/tests/datas/rope_s2_d96.safetensors"));
  const auto *cos_target_ptr =
      static_cast<const float *>(wm.get_tensor("cos").data);
  const auto *cos_ptr = cos.data<float>();
  for (size_t i = 0; i < 192; ++i) {
    EXPECT_FLOAT_EQ(cos_ptr[i], cos_target_ptr[i]);
  }
}

} // namespace tiny_llm
