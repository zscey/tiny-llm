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
    int64_t b = 3;
    int64_t h = 4;
    int64_t q = 2;
    int64_t d = 96;
    int64_t q_start = 1;
    int64_t q_end = 5;

    SafeTensorWeightManager wm(utils::BazelRunfile::RLocation(
        "tiny_llm/tests/datas/apply_rope_b3_h4_q2_d96_qs1qe5.safetensors"));

    Tensor position_ids({.type = DeviceType::kCpu, .id = 0}, DataType::kUint32,
                        {b, q});
    auto *position_ids_ptr = position_ids.data<uint32_t>();
    for (uint32_t i = 0; i < 6; ++i) {
      position_ids_ptr[i] = i & 1U;
    }
    position_ids = position_ids.to({.type = DeviceType::kCuda, .id = 0});

    Tensor dst({.type = DeviceType::kCpu, .id = 0}, DataType::kFloat32,
               {b, h, q_end, d});
    {
      size_t i{};
      auto *dst_ptr = dst.data<float>();
      for (int64_t cur_b = 0; cur_b < b; ++cur_b) {
        for (int64_t cur_h = 0; cur_h < h; ++cur_h) {
          auto *cur_ptr = dst_ptr + (((cur_b * h) + cur_h) * q_end * d);
          auto *cur_ptr_end = cur_ptr + (q_start * d);
          while (cur_ptr < cur_ptr_end) {
            *cur_ptr++ = 1.F;
          }
          cur_ptr_end = cur_ptr + (q * d);
          while (cur_ptr < cur_ptr_end) {
            *cur_ptr++ = static_cast<float>(i % 100) / 100.F;
            ++i;
          }
          cur_ptr_end = cur_ptr + ((q_end - q - q_start) * d);
          while (cur_ptr < cur_ptr_end) {
            *cur_ptr++ = 1.F;
          }
        }
      }
    }
    dst = dst.to({.type = DeviceType::kCuda, .id = 0});

    cuda::apply_rope_inplace(cos.data<float>(), sin.data<float>(),
                             position_ids.data<uint32_t>(), dst.data<float>(),
                             b, h, q, d, q_start, q_end);
    dst = dst.to({.type = DeviceType::kCpu, .id = 0});
    ThreadCudaContexts::Synchronize();

    const auto *dst_ptr = dst.data<float>();
    const auto *target_ptr =
        reinterpret_cast<const float *>(wm.get_tensor("res").data);
    for (size_t i = 0, i_end = b * h * q_end * d; i < i_end; ++i) {
      EXPECT_TRUE(std::abs(dst_ptr[i] - target_ptr[i]) < 1e-4F);
    }
  }

  { // Paged Rope
    int64_t total_queries = 6;
    int64_t h = 4;
    int64_t d = 96;

    Tensor position_ids({.type = DeviceType::kCpu, .id = 0}, DataType::kUint32,
                        {total_queries});
    {
      auto *position_ids_ptr = position_ids.data<uint32_t>();
      for (uint32_t i = 0; i < 6; ++i) {
        position_ids_ptr[i] = i & 1U;
      }
      position_ids = position_ids.to({.type = DeviceType::kCuda, .id = 0});
    }

    Tensor dst({.type = DeviceType::kCpu, .id = 0}, DataType::kFloat32,
               {2, h, 32, d});
    // 6 queries split to [2, 2, 2]
    Tensor block_table({.type = DeviceType::kCpu}, DataType::kUint32, {3, 1});
    {
      auto *block_table_ptr = block_table.data<uint32_t>();
      block_table_ptr[0] = 0;
      block_table_ptr[1] = 1;
      block_table_ptr[2] = 1;
      block_table = block_table.to({.type = DeviceType::kCuda, .id = 0});
    }
    Tensor seq_separator({.type = DeviceType::kCpu}, DataType::kUint32, {3});
    {
      auto *seq_separator_ptr = seq_separator.data<uint32_t>();
      seq_separator_ptr[0] = 0;
      seq_separator_ptr[1] = 2;
      seq_separator_ptr[2] = 4;
      seq_separator = seq_separator.to({.type = DeviceType::kCuda, .id = 0});
    }
    Tensor cache_offsets({.type = DeviceType::kCpu}, DataType::kUint32, {3});
    {
      auto *cache_offsets_ptr = cache_offsets.data<uint32_t>();
      cache_offsets_ptr[0] = 10;
      cache_offsets_ptr[1] = 15;
      cache_offsets_ptr[2] = 20;
      cache_offsets = cache_offsets.to({.type = DeviceType::kCuda, .id = 0});
    }

    {
      size_t i{};
      auto *dst_ptr = dst.data<float>();

      for (int64_t cur_h = 0; cur_h < h; ++cur_h) {
        auto *cur_ptr = dst_ptr + (((cur_h * 32) + 10) * d);
        auto *cur_ptr_end = cur_ptr + (2 * d);
        while (cur_ptr < cur_ptr_end) {
          *cur_ptr++ = static_cast<float>(i % 100) / 100.F;
          ++i;
        }
      }

      for (int64_t cur_h = 0; cur_h < h; ++cur_h) {
        auto *cur_ptr = dst_ptr + ((((h + cur_h) * 32) + 15) * d);
        auto *cur_ptr_end = cur_ptr + (2 * d);
        while (cur_ptr < cur_ptr_end) {
          *cur_ptr++ = static_cast<float>(i % 100) / 100.F;
          ++i;
        }
      }

      for (int64_t cur_h = 0; cur_h < h; ++cur_h) {
        auto *cur_ptr = dst_ptr + ((((h + cur_h) * 32) + 20) * d);
        auto *cur_ptr_end = cur_ptr + (2 * d);
        while (cur_ptr < cur_ptr_end) {
          *cur_ptr++ = static_cast<float>(i % 100) / 100.F;
          ++i;
        }
      }

      dst = dst.to({.type = DeviceType::kCuda, .id = 0});
    }

    cuda::apply_rope_inplace_paged(
        cos.data<float>(), sin.data<float>(), position_ids.data<uint32_t>(),
        dst.data<float>(), block_table.data<uint32_t>(),
        seq_separator.data<uint32_t>(), cache_offsets.data<uint32_t>(), 6, 3, 4,
        96, 1, 32);
    dst = dst.to({.type = DeviceType::kCpu, .id = 0});
    ThreadCudaContexts::Synchronize();

    const auto *dst_ptr = dst.data<float>();
    SafeTensorWeightManager wm(utils::BazelRunfile::RLocation(
        "tiny_llm/tests/datas/apply_rope_b3_h4_q2_d96_qs1qe5.safetensors"));
    const auto *target_ptr =
        reinterpret_cast<const float *>(wm.get_tensor("res").data);
    for (int64_t cur_h = 0; cur_h < h; ++cur_h) {
      {
        const auto *cur_dst_ptr = dst_ptr + (((cur_h * 32) + 10) * d);
        const auto *cur_target_ptr = target_ptr + ((cur_h * 5 + 1) * d);
        for (size_t i = 0; i < 192; ++i) {
          EXPECT_TRUE(std::abs(cur_dst_ptr[i] - cur_target_ptr[i]) < 1e-4F);
        }
      }
      {
        const auto *cur_dst_ptr = dst_ptr + ((((4 + cur_h) * 32) + 15) * d);
        const auto *cur_target_ptr = target_ptr + (((4 + cur_h) * 5 + 1) * d);
        for (size_t i = 0; i < 192; ++i) {
          EXPECT_TRUE(std::abs(cur_dst_ptr[i] - cur_target_ptr[i]) < 1e-4F);
        }
      }
      {
        const auto *cur_dst_ptr = dst_ptr + ((((4 + cur_h) * 32) + 20) * d);
        const auto *cur_target_ptr = target_ptr + (((8 + cur_h) * 5 + 1) * d);
        for (size_t i = 0; i < 192; ++i) {
          EXPECT_TRUE(std::abs(cur_dst_ptr[i] - cur_target_ptr[i]) < 1e-4F);
        }
      }
    }
  }
}

} // namespace tiny_llm
