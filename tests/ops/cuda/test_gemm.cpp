#include "tiny_llm/ops/cuda/gemm.hpp"
#include "tiny_llm/tensor/tensor.hpp"
#include "tiny_llm/weight_managers/safetensors/weight_manager.hpp"
#include "gtest/gtest.h"

namespace tiny_llm {
TEST(CudaOps, GemmManual) {
  int64_t m = 2;
  int64_t d = 3;
  int64_t n = 4;
  Device target_dev = {.type = DeviceType::kCuda, .id = 0};

  Tensor input({.type = DeviceType::kCpu}, DataType::kFloat32, {m, d});
  {
    auto *input_ptr = input.data<float>();
    for (size_t i = 0, i_end = static_cast<size_t>(m * d); i < i_end; ++i) {
      input_ptr[i] = static_cast<float>(i % 100);
    }
    input = input.to(target_dev);
  }

  Tensor weight({.type = DeviceType::kCpu}, DataType::kFloat32, {n, d});
  {
    auto *weight_ptr = weight.data<float>();
    for (size_t i = 0, i_end = static_cast<size_t>(n * d); i < i_end; ++i) {
      weight_ptr[i] = static_cast<float>(i % 100);
    }
    weight = weight.to(target_dev);
  }

  Tensor dst_plain(target_dev, DataType::kFloat32, {m, n});
  cuda::gemm_row_major_plain(input.data<float>(), weight.data<float>(), nullptr,
                             dst_plain.data<float>(), m, d, n);
  auto cpu_dst_plain = dst_plain.to({.type = DeviceType::kCpu});

  Tensor dst(target_dev, DataType::kFloat32, {m, n});
  cuda::gemm_row_major(input.data<float>(), weight.data<float>(), nullptr,
                       dst.data<float>(), m, d, n);
  auto cpu_dst = dst.to({.type = DeviceType::kCpu});

  std::vector<float> target_res{5.F, 14.F, 23.F, 32.F, 14.F, 50.F, 86.F, 122.F};
  ThreadCudaContexts::Synchronize();

  const auto *dst_plain_ptr = cpu_dst_plain.data<float>();
  const auto *dst_ptr = cpu_dst.data<float>();
  for (size_t i = 0, i_end = static_cast<size_t>(m * n); i < i_end; ++i) {
    EXPECT_FLOAT_EQ(dst_plain_ptr[i], target_res.at(i));
    EXPECT_FLOAT_EQ(dst_ptr[i], target_res.at(i));
  }
}

TEST(CudaOps, Gemm) {
  int64_t m = 127;
  int64_t d = 255;
  int64_t n = 353;
  Device target_dev = {.type = DeviceType::kCuda, .id = 0};

  Tensor input({.type = DeviceType::kCpu}, DataType::kFloat32, {m, d});
  {
    auto *input_ptr = input.data<float>();
    for (size_t i = 0, i_end = static_cast<size_t>(m * d); i < i_end; ++i) {
      input_ptr[i] = static_cast<float>(i % 100);
    }
    input = input.to(target_dev);
  }

  Tensor weight({.type = DeviceType::kCpu}, DataType::kFloat32, {n, d});
  {
    auto *weight_ptr = weight.data<float>();
    for (size_t i = 0, i_end = static_cast<size_t>(n * d); i < i_end; ++i) {
      weight_ptr[i] = static_cast<float>(i % 100);
    }
    weight = weight.to(target_dev);
  }

  Tensor dst_plain(target_dev, DataType::kFloat32, {m, n});
  cuda::gemm_row_major_plain(input.data<float>(), weight.data<float>(), nullptr,
                             dst_plain.data<float>(), m, d, n);
  auto cpu_dst_plain = dst_plain.to({.type = DeviceType::kCpu});

  Tensor dst(target_dev, DataType::kFloat32, {m, n});
  cuda::gemm_row_major(input.data<float>(), weight.data<float>(), nullptr,
                       dst.data<float>(), m, d, n);
  auto cpu_dst = dst.to({.type = DeviceType::kCpu});

  ThreadCudaContexts::Synchronize();

  SafeTensorWeightManager wm("/home/haol/code-dir/attention-test/"
                             "gemm_m127d255n353.safetensors");
  const auto *dst_plain_ptr = cpu_dst_plain.data<float>();
  const auto *dst_ptr = cpu_dst.data<float>();
  const auto *target_ptr =
      reinterpret_cast<const float *>(wm.get_tensor("res").data);
  for (size_t i = 0, i_end = static_cast<size_t>(m * n); i < i_end; ++i) {
    EXPECT_FLOAT_EQ(dst_plain_ptr[i], target_ptr[i]);
    EXPECT_FLOAT_EQ(dst_ptr[i], target_ptr[i]);
  }
}
} // namespace tiny_llm
