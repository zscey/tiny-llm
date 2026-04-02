#include "tiny_llm/ops/cuda/gemm.hpp"
#include "tiny_llm/tensor/tensor.hpp"
#include "tiny_llm/utils/runfile.hpp"
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

  Tensor dst(target_dev, DataType::kFloat32, {m, n});
  cuda::gemm_row_major(input.data<float>(), weight.data<float>(), nullptr,
                       dst.data<float>(), m, d, n);
  auto cpu_dst = dst.to({.type = DeviceType::kCpu});

  std::vector<float> target_res{5.F, 14.F, 23.F, 32.F, 14.F, 50.F, 86.F, 122.F};
  ThreadCudaContexts::Synchronize();

  const auto *dst_ptr = cpu_dst.data<float>();
  for (size_t i = 0, i_end = static_cast<size_t>(m * n); i < i_end; ++i) {
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

  Tensor dst(target_dev, DataType::kFloat32, {m, n});
  cuda::gemm_row_major(input.data<float>(), weight.data<float>(), nullptr,
                       dst.data<float>(), m, d, n);
  auto cpu_dst = dst.to({.type = DeviceType::kCpu});

  ThreadCudaContexts::Synchronize();

  SafeTensorWeightManager wm(utils::BazelRunfile::RLocation(
      "tiny_llm/tests/datas/gemm_m127d255n353.safetensors"));
  const auto *dst_ptr = cpu_dst.data<float>();
  const auto *target_ptr =
      reinterpret_cast<const float *>(wm.get_tensor("res").data);
  for (size_t i = 0, i_end = static_cast<size_t>(m * n); i < i_end; ++i) {
    EXPECT_FLOAT_EQ(dst_ptr[i], target_ptr[i]);
  }
}

TEST(CudaOps, GemmLT) {
  int64_t b = 2;
  int64_t q = 61;
  int64_t d = 255;
  int64_t out_head = 4;
  int64_t out_dim = 88;
  int64_t q_start = 3;
  int64_t q_end = q_start + q + 13;

  Device target_dev = {.type = DeviceType::kCuda, .id = 0};

  Tensor input({.type = DeviceType::kCpu}, DataType::kFloat32, {b, q, d});
  {
    auto *input_ptr = input.data<float>();
    for (size_t i = 0, i_end = static_cast<size_t>(b * q * d); i < i_end; ++i) {
      input_ptr[i] = static_cast<float>(i % 100);
    }
    input = input.to(target_dev);
  }

  Tensor weight({.type = DeviceType::kCpu}, DataType::kFloat32,
                {out_head * out_dim, d});
  {
    auto *weight_ptr = weight.data<float>();
    for (size_t i = 0, i_end = static_cast<size_t>(out_head * out_dim * d);
         i < i_end; ++i) {
      weight_ptr[i] = static_cast<float>(i % 100);
    }
    weight = weight.to(target_dev);
  }

  Tensor dst({.type = DeviceType::kCpu}, DataType::kFloat32,
             {b, out_head, q_end, out_dim});
  {
    auto *dst_ptr = dst.data<float>();
    for (size_t i = 0,
                i_end = static_cast<size_t>(b * out_head * q_end * out_dim);
         i < i_end; ++i) {
      dst_ptr[i] = 1.F;
    }
    dst = dst.to(target_dev);
  }
  cuda::gemm_row_major_lt(input.data<float>(), weight.data<float>(), nullptr,
                          dst.data<float>(), b, q, d, out_head, out_dim,
                          q_start, q_end);
  auto cpu_dst = dst.to({.type = DeviceType::kCpu});

  ThreadCudaContexts::Synchronize();

  SafeTensorWeightManager wm(utils::BazelRunfile::RLocation(
      "tiny_llm/tests/datas/gemm_b2q61d255oh4od88qs3qe77.safetensors"));
  const auto *dst_ptr = cpu_dst.data<float>();
  const auto *target_ptr =
      reinterpret_cast<const float *>(wm.get_tensor("res").data);
  for (size_t i = 0,
              i_end = static_cast<size_t>(b * out_head * q_end * out_dim);
       i < i_end; ++i) {
    EXPECT_FLOAT_EQ(dst_ptr[i], target_ptr[i]);
  }
}

TEST(CudaOps, GemmTL) {
  int64_t b = 2;
  int64_t h = 4;
  int64_t q = 127;
  int64_t d = 61;
  int64_t out_d = 313;

  Device target_dev = {.type = DeviceType::kCuda, .id = 0};

  Tensor input({.type = DeviceType::kCpu}, DataType::kFloat32, {b, h, q, d});
  {
    auto *input_ptr = input.data<float>();
    for (size_t i = 0, i_end = static_cast<size_t>(b * h * q * d); i < i_end;
         ++i) {
      input_ptr[i] = static_cast<float>(i % 100);
    }
    input = input.to(target_dev);
  }

  Tensor weight({.type = DeviceType::kCpu}, DataType::kFloat32, {out_d, h * d});
  {
    auto *weight_ptr = weight.data<float>();
    for (size_t i = 0, i_end = static_cast<size_t>(out_d * h * d); i < i_end;
         ++i) {
      weight_ptr[i] = static_cast<float>(i % 100);
    }
    weight = weight.to(target_dev);
  }

  Tensor dst(target_dev, DataType::kFloat32, {b, q, out_d});
  cuda::gemm_row_major_tl(input.data<float>(), weight.data<float>(), nullptr,
                          dst.data<float>(), b, h, q, d, out_d);
  auto cpu_dst = dst.to({.type = DeviceType::kCpu});

  ThreadCudaContexts::Synchronize();

  SafeTensorWeightManager wm(utils::BazelRunfile::RLocation(
      "tiny_llm/tests/datas/gemm_b2h4q127d61od313.safetensors"));
  const auto *dst_ptr = cpu_dst.data<float>();
  const auto *target_ptr =
      reinterpret_cast<const float *>(wm.get_tensor("res").data);
  for (size_t i = 0, i_end = static_cast<size_t>(b * q * out_d); i < i_end;
       ++i) {
    EXPECT_FLOAT_EQ(dst_ptr[i], target_ptr[i]);
  }
}

TEST(CudaOps, GemmL) {
  int64_t b = 2;
  int64_t q_start = 10;
  int64_t q = 40;
  int64_t q_end = q_start + q + 20;
  int64_t d = 255;
  int64_t n = 353;
  Device target_dev = {.type = DeviceType::kCuda, .id = 0};

  Tensor input({.type = DeviceType::kCpu}, DataType::kFloat32, {b, q_end, d});
  {
    auto *input_ptr = input.data<float>();
    size_t i{};
    for (int64_t cur_b = 0; cur_b < b; ++cur_b) {
      auto *cur_ptr = input_ptr + (((cur_b * q_end) + q_start) * d);
      auto *cur_ptr_end = cur_ptr + (q * d);
      while (cur_ptr < cur_ptr_end) {
        *cur_ptr++ = static_cast<float>(i % 100);
        ++i;
      }
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

  Tensor dst(target_dev, DataType::kFloat32, {b, q, n});
  cuda::gemm_row_major_l(input.data<float>(), weight.data<float>(), nullptr,
                         dst.data<float>(), b, q, d, n, q_start, q_end);
  auto cpu_dst = dst.to({.type = DeviceType::kCpu});

  ThreadCudaContexts::Synchronize();

  SafeTensorWeightManager wm(utils::BazelRunfile::RLocation(
      "tiny_llm/tests/datas/gemm_m127d255n353.safetensors"));
  const auto *dst_ptr = cpu_dst.data<float>();
  const auto *target_ptr =
      reinterpret_cast<const float *>(wm.get_tensor("res").data);
  for (size_t i = 0, i_end = static_cast<size_t>(b) * q * n; i < i_end; ++i) {
    EXPECT_FLOAT_EQ(dst_ptr[i], target_ptr[i]);
  }
}
} // namespace tiny_llm
