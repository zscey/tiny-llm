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
    query_ptr[i] = static_cast<float>(i % 100) / 100.F;
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
          cur_ptr[shift] = static_cast<float>((i + 1) % 100) / 100.F;
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
          cur_ptr[shift] = static_cast<float>((i + 2) % 100) / 100.F;
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
    query_ptr[i] = static_cast<float>(i % 100) / 100.F;
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
          cur_ptr[shift] = static_cast<float>((i + 1) % 100) / 100.F;
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
          cur_ptr[shift] = static_cast<float>((i + 2) % 100) / 100.F;
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

TEST(CudaOps, FlashAttnNoMaskPaged) {
  int64_t batch = 2;
  int64_t q_length = 107;
  int64_t q_head = 4;
  int64_t kv_length = 123;
  int64_t kv_head = 2;
  int64_t dim = 61;

  Tensor query({.type = DeviceType::kCpu, .id = 0}, DataType::kFloat32,
               {q_head, batch * q_length, dim});
  {
    auto *query_ptr = query.data<float>();
    size_t i{};
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t h = 0; h < q_head; ++h) {
        auto *cur_ptr = query_ptr + (((h * batch) + b) * q_length * dim);
        for (int64_t qd = 0, qd_end = q_length * dim; qd < qd_end; ++qd) {
          cur_ptr[qd] = static_cast<float>(i % 100) / 100.F;
          ++i;
        }
      }
    }
    query = query.to({.type = DeviceType::kCuda, .id = 0});
  }

  Tensor key({.type = DeviceType::kCpu, .id = 0}, DataType::kFloat32,
             {8, kv_head, 32, dim});
  auto *key_ptr = key.data<float>();
  Tensor value({.type = DeviceType::kCpu, .id = 0}, DataType::kFloat32,
               {8, kv_head, 32, dim});
  auto *value_ptr = value.data<float>();
  {
    size_t i{};
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t h = 0; h < kv_head; ++h) {
        for (int64_t q = 0; q < kv_length; ++q) {
          auto shift = (((b * 4 + q / 32) * kv_head + h) * 32 + (q % 32)) * dim;
          auto *cur_key_ptr = key_ptr + shift;
          auto *cur_value_ptr = value_ptr + shift;
          for (int64_t d = 0; d < dim; ++d) {
            cur_key_ptr[d] = static_cast<float>((i + 1) % 100) / 100.F;
            cur_value_ptr[d] = static_cast<float>((i + 2) % 100) / 100.F;
            ++i;
          }
        }
      }
    }
    key = key.to({.type = DeviceType::kCuda, .id = 0});
    value = value.to({.type = DeviceType::kCuda, .id = 0});
  }

  Tensor dst({.type = DeviceType::kCuda, .id = 0}, DataType::kFloat32,
             {q_head, batch * q_length, dim});
  Tensor q_separator({.type = DeviceType::kCpu}, DataType::kUint32, {3});
  {
    auto *q_separator_ptr = q_separator.data<uint32_t>();
    q_separator_ptr[0] = 0;
    q_separator_ptr[1] = 107;
    q_separator_ptr[2] = 214;
    q_separator = q_separator.to({.type = DeviceType::kCuda, .id = 0});
  }
  Tensor block_table({.type = DeviceType::kCpu}, DataType::kUint32, {2, 4});
  {
    auto *block_table_ptr = block_table.data<uint32_t>();
    for (uint32_t i = 0; i < 8; ++i) {
      block_table_ptr[i] = i;
    }
    block_table = block_table.to({.type = DeviceType::kCuda, .id = 0});
  }
  Tensor kv_lens({.type = DeviceType::kCpu}, DataType::kUint32, {2});
  {
    auto *kv_lens_ptr = kv_lens.data<uint32_t>();
    kv_lens_ptr[0] = 123;
    kv_lens_ptr[1] = 123;
    kv_lens = kv_lens.to({.type = DeviceType::kCuda, .id = 0});
  }
  cuda::flash_attn_paged(query.data<float>(), key.data<float>(),
                         value.data<float>(), dst.data<float>(),
                         q_separator.data<uint32_t>(),
                         block_table.data<uint32_t>(), kv_lens.data<uint32_t>(),
                         2, 4, 2, 61, 4, 32, 107, cuda::AttentionType::kNoMask);
  dst = dst.to({.type = DeviceType::kCpu, .id = 0});
  ThreadCudaContexts::Synchronize();

  SafeTensorWeightManager wm(utils::BazelRunfile::RLocation(
      "tiny_llm/tests/datas/attn_res_b2q107h4k123h2d61.safetensors"));
  const auto *dst_ptr = dst.data<float>();
  const auto *target_ptr =
      reinterpret_cast<const float *>(wm.get_tensor("res").data);

  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t h = 0; h < q_head; ++h) {
      const auto *cur_ptr = dst_ptr + (((h * batch) + b) * q_length * dim);
      for (int64_t qd = 0, qd_end = q_length * dim; qd < qd_end; ++qd) {
        EXPECT_TRUE(std::abs(cur_ptr[qd] - *target_ptr) < 1e-5F);
        ++target_ptr;
      }
    }
  }
}

TEST(CudaOps, FlashAttnCausalMaskPaged) {
  int64_t batch = 2;
  int64_t q_length = 107;
  int64_t q_head = 4;
  int64_t kv_length = q_length;
  int64_t kv_head = 2;
  int64_t dim = 61;

  Tensor query({.type = DeviceType::kCpu, .id = 0}, DataType::kFloat32,
               {q_head, batch * q_length, dim});
  {
    auto *query_ptr = query.data<float>();
    size_t i{};
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t h = 0; h < q_head; ++h) {
        auto *cur_ptr = query_ptr + (((h * batch) + b) * q_length * dim);
        for (int64_t qd = 0, qd_end = q_length * dim; qd < qd_end; ++qd) {
          cur_ptr[qd] = static_cast<float>(i % 100) / 100.F;
          ++i;
        }
      }
    }
    query = query.to({.type = DeviceType::kCuda, .id = 0});
  }

  Tensor key({.type = DeviceType::kCpu, .id = 0}, DataType::kFloat32,
             {8, kv_head, 32, dim});
  auto *key_ptr = key.data<float>();
  Tensor value({.type = DeviceType::kCpu, .id = 0}, DataType::kFloat32,
               {8, kv_head, 32, dim});
  auto *value_ptr = value.data<float>();
  {
    size_t i{};
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t h = 0; h < kv_head; ++h) {
        for (int64_t q = 0; q < kv_length; ++q) {
          auto shift = (((b * 4 + q / 32) * kv_head + h) * 32 + (q % 32)) * dim;
          auto *cur_key_ptr = key_ptr + shift;
          auto *cur_value_ptr = value_ptr + shift;
          for (int64_t d = 0; d < dim; ++d) {
            cur_key_ptr[d] = static_cast<float>((i + 1) % 100) / 100.F;
            cur_value_ptr[d] = static_cast<float>((i + 2) % 100) / 100.F;
            ++i;
          }
        }
      }
    }
    key = key.to({.type = DeviceType::kCuda, .id = 0});
    value = value.to({.type = DeviceType::kCuda, .id = 0});
  }

  Tensor dst({.type = DeviceType::kCuda, .id = 0}, DataType::kFloat32,
             {q_head, batch * q_length, dim});
  Tensor q_separator({.type = DeviceType::kCpu}, DataType::kUint32, {3});
  {
    auto *q_separator_ptr = q_separator.data<uint32_t>();
    q_separator_ptr[0] = 0;
    q_separator_ptr[1] = 107;
    q_separator_ptr[2] = 214;
    q_separator = q_separator.to({.type = DeviceType::kCuda, .id = 0});
  }
  Tensor block_table({.type = DeviceType::kCpu}, DataType::kUint32, {2, 4});
  {
    auto *block_table_ptr = block_table.data<uint32_t>();
    for (uint32_t i = 0; i < 8; ++i) {
      block_table_ptr[i] = i;
    }
    block_table = block_table.to({.type = DeviceType::kCuda, .id = 0});
  }
  Tensor kv_lens({.type = DeviceType::kCpu}, DataType::kUint32, {2});
  {
    auto *kv_lens_ptr = kv_lens.data<uint32_t>();
    kv_lens_ptr[0] = 107;
    kv_lens_ptr[1] = 107;
    kv_lens = kv_lens.to({.type = DeviceType::kCuda, .id = 0});
  }

  cuda::flash_attn_paged(
      query.data<float>(), key.data<float>(), value.data<float>(),
      dst.data<float>(), q_separator.data<uint32_t>(),
      block_table.data<uint32_t>(), kv_lens.data<uint32_t>(), 2, 4, 2, 61, 4,
      32, 107, cuda::AttentionType::kCausalMask);
  dst = dst.to({.type = DeviceType::kCpu, .id = 0});
  ThreadCudaContexts::Synchronize();

  SafeTensorWeightManager wm(utils::BazelRunfile::RLocation(
      "tiny_llm/tests/datas/causal_attn_res_b2l107q4k2d61.safetensors"));
  const auto *dst_ptr = dst.data<float>();
  const auto *target_ptr =
      reinterpret_cast<const float *>(wm.get_tensor("res").data);

  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t h = 0; h < q_head; ++h) {
      const auto *cur_ptr = dst_ptr + (((h * batch) + b) * q_length * dim);
      for (int64_t qd = 0, qd_end = q_length * dim; qd < qd_end; ++qd) {
        EXPECT_TRUE(std::abs(cur_ptr[qd] - *target_ptr) < 1e-5F);
        ++target_ptr;
      }
    }
  }
}
} // namespace tiny_llm
