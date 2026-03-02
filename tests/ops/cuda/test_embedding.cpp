#include "tiny_llm/ops/cuda/embedding.hpp"
#include "tiny_llm/tensor/tensor.hpp"
#include "gtest/gtest.h"

namespace tiny_llm {
TEST(CudaOps, Embedding) {
  int64_t emb_size = 10;
  int64_t dim = 2011;

  Tensor emb_weight({.type = DeviceType::kCpu, .id = 0}, DataType::kFloat32,
                    {emb_size, dim});
  auto *emb_weight_ptr = emb_weight.data<float>();
  for (uint32_t i = 0, i_end = static_cast<uint32_t>(emb_size * dim); i < i_end;
       ++i) {
    emb_weight_ptr[i] = static_cast<float>(i);
  }
  auto gpu_emb_weight = emb_weight.to({.type = DeviceType::kCuda, .id = 0});

  Tensor src({.type = DeviceType::kCpu, .id = 0}, DataType::kUint32, {2, 3});
  auto *src_ptr = src.data<uint32_t>();
  src_ptr[0] = 2;
  src_ptr[1] = 3;
  src_ptr[2] = 1;
  src_ptr[3] = 7;
  src_ptr[4] = 9;
  src_ptr[5] = 0;
  auto gpu_src = src.to({.type = DeviceType::kCuda, .id = 0});

  Tensor dst({.type = DeviceType::kCuda, .id = 0}, DataType::kFloat32,
             {2, 3, dim});
  cuda::embedding(gpu_src.data<uint32_t>(), emb_weight.data<float>(),
                  dst.data<float>(), dim, 6);
  dst = dst.to({.type = DeviceType::kCpu, .id = 0});

  ThreadCudaContexts::Synchronize();

  for (size_t i = 0; i < 6; ++i) {
    const auto *target_ptr = emb_weight_ptr + (src_ptr[i] * dim);
    const auto *dst_ptr = dst.data<float>() + (i * dim);
    for (int64_t j = 0; j < dim; ++j) {
      EXPECT_FLOAT_EQ(target_ptr[j], dst_ptr[j]);
    }
  }
}

} // namespace tiny_llm
