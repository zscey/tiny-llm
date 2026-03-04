#include "tiny_llm/ops/cuda/rope.hpp"
#include "tiny_llm/tensor/tensor.hpp"
#include "tiny_llm/utils/runfile.hpp"
#include "tiny_llm/weight_managers/safetensors/weight_manager.hpp"
#include "gtest/gtest.h"

namespace tiny_llm {
TEST(CudaOps, Rope) {
  Tensor cos({.type = DeviceType::kCuda, .id = 0}, DataType::kFloat32, {2, 48},
             true);
  Tensor sin({.type = DeviceType::kCuda, .id = 0}, DataType::kFloat32, {2, 48},
             true);
  cuda::rope(cos.data<float>(), sin.data<float>(), 2, 96, 10000);

  {
    auto cpu_cos = cos.to({.type = DeviceType::kCpu, .id = 0});
    auto cpu_sin = sin.to({.type = DeviceType::kCpu, .id = 0});
    ThreadCudaContexts::Synchronize();

    SafeTensorWeightManager wm(utils::BazelRunfile::RLocation(
        "tiny_llm/tests/datas/rope_s2_d96.safetensors"));
    const auto *cos_target_ptr =
        static_cast<const float *>(wm.get_tensor("cos").data);
    const auto *cos_ptr = cpu_cos.data<float>();
    const auto *sin_target_ptr =
        static_cast<const float *>(wm.get_tensor("sin").data);
    const auto *sin_ptr = cpu_sin.data<float>();
    for (size_t i = 0; i < 2; ++i) {
      for (size_t j = 0; j < 48; ++j) {
        auto dst_shift = (i * 48) + j;
        auto target_shift = (i * 96) + j;

        EXPECT_FLOAT_EQ(cos_ptr[dst_shift], cos_target_ptr[target_shift]);
        EXPECT_FLOAT_EQ(cos_ptr[dst_shift], cos_target_ptr[target_shift + 48]);
        EXPECT_FLOAT_EQ(sin_ptr[dst_shift], sin_target_ptr[target_shift]);
        EXPECT_FLOAT_EQ(sin_ptr[dst_shift], sin_target_ptr[target_shift + 48]);
      }
    }
  }

  {
    SafeTensorWeightManager wm(utils::BazelRunfile::RLocation(
        "tiny_llm/tests/datas/apply_rope_s2_d96.safetensors"));

    Tensor position_ids({.type = DeviceType::kCpu, .id = 0}, DataType::kUint32,
                        {2});
    auto *position_ids_ptr = position_ids.data<uint32_t>();
    position_ids_ptr[0] = 0;
    position_ids_ptr[1] = 1;
    position_ids = position_ids.to({.type = DeviceType::kCuda, .id = 0});

    Tensor dst({.type = DeviceType::kCpu, .id = 0}, DataType::kFloat32,
               {1, 2, 288});
    {
      auto *dst_ptr = dst.data<float>();
      for (size_t i = 0; i < 576; ++i) {
        dst_ptr[i] = static_cast<float>(i);
      }
    }
    dst = dst.to({.type = DeviceType::kCuda, .id = 0});

    cuda::apply_rope_inplace(cos.data<float>(), sin.data<float>(),
                             position_ids.data<uint32_t>(), dst.data<float>(),
                             2, 3, 96);
    dst = dst.to({.type = DeviceType::kCpu, .id = 0});
    ThreadCudaContexts::Synchronize();

    const auto *dst_ptr = dst.data<float>();
    const auto *target_ptr =
        reinterpret_cast<const float *>(wm.get_tensor("res").data);
    for (size_t i = 0; i < 576; ++i) {
      EXPECT_TRUE(std::abs(dst_ptr[i] - target_ptr[i]) < 1e-4F);
    }
  }
}

} // namespace tiny_llm
