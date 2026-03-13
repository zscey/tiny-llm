#include "tiny_llm/ops/cuda/flash_attention.hpp"
#include "tiny_llm/tensor/tensor.hpp"
#include "tiny_llm/utils/runfile.hpp"
#include "tiny_llm/weight_managers/safetensors/weight_manager.hpp"
#include "gtest/gtest.h"

namespace tiny_llm {
TEST(CudaOps, Func) {
  uint32_t batch = 2;
  uint32_t q_length = 1013;
  uint32_t q_head = 4;
  uint32_t kv_length = 2019;
  uint32_t kv_head = 2;
  uint32_t dim = 61;

  Tensor query({.type = DeviceType::kCpu, .id = 0}, DataType::kFloat32,
               {batch, q_length, static_cast<int64_t>(q_head * dim)});
  auto *query_ptr = query.data<float>();
  for (size_t i = 0,
              i_end = static_cast<size_t>(batch) * q_length * q_head * dim;
       i < i_end; ++i) {
    query_ptr[i] = static_cast<float>(i % 100) / 100;
  }
  query = query.to({.type = DeviceType::kCuda, .id = 0});

  Tensor key({.type = DeviceType::kCpu, .id = 0}, DataType::kFloat32,
             {batch, kv_length, static_cast<int64_t>(kv_head * dim)});
  auto *key_ptr = key.data<float>();
  for (size_t i = 0,
              i_end = static_cast<size_t>(batch) * kv_length * kv_head * dim;
       i < i_end; ++i) {
    key_ptr[i] = static_cast<float>((i + 1) % 100) / 100;
  }
  key = key.to({.type = DeviceType::kCuda, .id = 0});

  Tensor value({.type = DeviceType::kCpu, .id = 0}, DataType::kFloat32,
               {batch, kv_length, static_cast<int64_t>(kv_head * dim)});
  auto *value_ptr = value.data<float>();
  for (size_t i = 0,
              i_end = static_cast<size_t>(batch) * kv_length * kv_head * dim;
       i < i_end; ++i) {
    value_ptr[i] = static_cast<float>((i + 2) % 100) / 100;
  }
  value = value.to({.type = DeviceType::kCuda, .id = 0});

  {
    Tensor dst({.type = DeviceType::kCuda, .id = 0}, DataType::kFloat32,
               {batch, q_length, static_cast<int64_t>(q_head * dim)});
    cuda::flash_attn(query.data<float>(), key.data<float>(),
                     value.data<float>(), dst.data<float>(), batch, q_length,
                     kv_length, dim, q_head, kv_head);
    dst = dst.to({.type = DeviceType::kCpu, .id = 0});
    ThreadCudaContexts::Synchronize();

    SafeTensorWeightManager wm("/home/haol/code-dir/attention-test/"
                               "attn_res_b2q1013h4k2019h2d61.safetensors");
    const auto *dst_ptr = dst.data<float>();
    const auto *target_ptr =
        reinterpret_cast<const float *>(wm.get_tensor("res").data);
    for (size_t i = 0,
                i_end = static_cast<size_t>(batch) * q_length * q_head * dim;
         i < i_end; ++i) {
      EXPECT_TRUE(std::abs(dst_ptr[i] - target_ptr[i]) < 1e-5F) << i;
    }
  }
}

} // namespace tiny_llm
