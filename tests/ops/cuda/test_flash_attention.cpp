#include "tiny_llm/ops/cuda/flash_attention.hpp"
#include "tiny_llm/tensor/tensor.hpp"
#include "tiny_llm/utils/runfile.hpp"
#include "tiny_llm/weight_managers/safetensors/weight_manager.hpp"
#include "gtest/gtest.h"

namespace tiny_llm {
TEST(CudaOps, FlashAttnNoMask) {
  uint32_t batch = 2;
  uint32_t q_length = 107;
  uint32_t q_head = 4;
  uint32_t kv_length = 123;
  uint32_t kv_head = 2;
  uint32_t dim = 61;

  uint32_t kv_padding = 10;

  Tensor query({.type = DeviceType::kCpu, .id = 0}, DataType::kFloat32,
               {batch, q_head, q_length, dim});
  auto *query_ptr = query.data<float>();
  for (size_t i = 0,
              i_end = static_cast<size_t>(batch) * q_head * q_length * dim;
       i < i_end; ++i) {
    query_ptr[i] = static_cast<float>(i % 100) / 100;
  }
  query = query.to({.type = DeviceType::kCuda, .id = 0});

  Tensor key({.type = DeviceType::kCpu, .id = 0}, DataType::kFloat32,
             {batch, kv_head, kv_length + kv_padding, dim});
  auto *key_ptr = key.data<float>();
  {
    size_t i{};
    for (size_t b = 0; b < batch; ++b) {
      for (size_t h = 0; h < kv_head; ++h) {
        auto *cur_ptr =
            key_ptr + ((((b * kv_head) + h) * (kv_length + kv_padding)) * dim);
        for (size_t shift = 0, shift_end = static_cast<size_t>(kv_length) * dim;
             shift < shift_end; ++shift) {
          cur_ptr[shift] = static_cast<float>((i + 1) % 100) / 100;
          ++i;
        }
      }
    }
  }
  key = key.to({.type = DeviceType::kCuda, .id = 0});

  Tensor value({.type = DeviceType::kCpu, .id = 0}, DataType::kFloat32,
               {batch, kv_head, kv_length + kv_padding, dim});
  auto *value_ptr = value.data<float>();
  {
    size_t i{};
    for (size_t b = 0; b < batch; ++b) {
      for (size_t h = 0; h < kv_head; ++h) {
        auto *cur_ptr =
            value_ptr +
            ((((b * kv_head) + h) * (kv_length + kv_padding)) * dim);
        for (size_t shift = 0, shift_end = static_cast<size_t>(kv_length) * dim;
             shift < shift_end; ++shift) {
          cur_ptr[shift] = static_cast<float>((i + 2) % 100) / 100;
          ++i;
        }
      }
    }
  }
  value = value.to({.type = DeviceType::kCuda, .id = 0});

  {
    Tensor dst({.type = DeviceType::kCuda, .id = 0}, DataType::kFloat32,
               {batch, q_length, static_cast<int64_t>(q_head * dim)});
    cuda::flash_attn(query.data<float>(), key.data<float>(),
                     value.data<float>(), dst.data<float>(), batch, q_head,
                     kv_head, q_length, kv_length, dim, kv_length + kv_padding,
                     cuda::AttentionType::kNoMask);
    dst = dst.to({.type = DeviceType::kCpu, .id = 0});
    ThreadCudaContexts::Synchronize();

    SafeTensorWeightManager wm(utils::BazelRunfile::RLocation(
        "tiny_llm/tests/datas/attn_res_b2q107h4k123h2d61.safetensors"));
    const auto *dst_ptr = dst.data<float>();
    const auto *target_ptr =
        reinterpret_cast<const float *>(wm.get_tensor("res").data);
    for (size_t i = 0,
                i_end = static_cast<size_t>(batch) * q_length * q_head * dim;
         i < i_end; ++i) {
      EXPECT_TRUE(std::abs(dst_ptr[i] - target_ptr[i]) < 1e-5F);
    }
  }
}

TEST(CudaOps, FlashAttnCausalMask) {
  uint32_t batch = 2;
  uint32_t q_length = 107;
  uint32_t q_head = 4;
  uint32_t kv_length = q_length;
  uint32_t kv_head = 2;
  uint32_t dim = 61;

  uint32_t kv_padding = 10;

  Tensor query({.type = DeviceType::kCpu, .id = 0}, DataType::kFloat32,
               {batch, q_head, q_length, dim});
  auto *query_ptr = query.data<float>();
  for (size_t i = 0,
              i_end = static_cast<size_t>(batch) * q_head * q_length * dim;
       i < i_end; ++i) {
    query_ptr[i] = static_cast<float>(i % 100) / 100;
  }
  query = query.to({.type = DeviceType::kCuda, .id = 0});

  Tensor key({.type = DeviceType::kCpu, .id = 0}, DataType::kFloat32,
             {batch, kv_head, kv_length + kv_padding, dim});
  auto *key_ptr = key.data<float>();
  {
    size_t i{};
    for (size_t b = 0; b < batch; ++b) {
      for (size_t h = 0; h < kv_head; ++h) {
        auto *cur_ptr =
            key_ptr + ((((b * kv_head) + h) * (kv_length + kv_padding)) * dim);
        for (size_t shift = 0, shift_end = static_cast<size_t>(kv_length) * dim;
             shift < shift_end; ++shift) {
          cur_ptr[shift] = static_cast<float>((i + 1) % 100) / 100;
          ++i;
        }
      }
    }
  }
  key = key.to({.type = DeviceType::kCuda, .id = 0});

  Tensor value({.type = DeviceType::kCpu, .id = 0}, DataType::kFloat32,
               {batch, kv_head, kv_length + kv_padding, dim});
  auto *value_ptr = value.data<float>();
  {
    size_t i{};
    for (size_t b = 0; b < batch; ++b) {
      for (size_t h = 0; h < kv_head; ++h) {
        auto *cur_ptr =
            value_ptr +
            ((((b * kv_head) + h) * (kv_length + kv_padding)) * dim);
        for (size_t shift = 0, shift_end = static_cast<size_t>(kv_length) * dim;
             shift < shift_end; ++shift) {
          cur_ptr[shift] = static_cast<float>((i + 2) % 100) / 100;
          ++i;
        }
      }
    }
  }
  value = value.to({.type = DeviceType::kCuda, .id = 0});

  {
    Tensor dst({.type = DeviceType::kCuda, .id = 0}, DataType::kFloat32,
               {batch, q_length, static_cast<int64_t>(q_head * dim)});
    cuda::flash_attn(query.data<float>(), key.data<float>(),
                     value.data<float>(), dst.data<float>(), batch, q_head,
                     kv_head, q_length, kv_length, dim, kv_length + kv_padding,
                     cuda::AttentionType::kCausalMask);
    dst = dst.to({.type = DeviceType::kCpu, .id = 0});
    ThreadCudaContexts::Synchronize();

    SafeTensorWeightManager wm(utils::BazelRunfile::RLocation(
        "tiny_llm/tests/datas/causal_attn_res_b2l107q4k2d61.safetensors"));
    const auto *dst_ptr = dst.data<float>();
    const auto *target_ptr =
        reinterpret_cast<const float *>(wm.get_tensor("res").data);
    for (size_t i = 0,
                i_end = static_cast<size_t>(batch) * q_length * q_head * dim;
         i < i_end; ++i) {
      EXPECT_TRUE(std::abs(dst_ptr[i] - target_ptr[i]) < 1e-5F);
    }
  }
}
} // namespace tiny_llm
