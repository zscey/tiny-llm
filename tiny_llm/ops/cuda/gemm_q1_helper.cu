#include "cccl/cuda/warp"
#include "cub/cub.cuh"
#include "tiny_llm/common/checks.hpp"
#include "tiny_llm/common/exception.hpp"
#include "tiny_llm/ops/cuda/gemm_q1_helper.hpp"

namespace tiny_llm::cuda::q1 {
namespace {
constexpr uint32_t kWarpSize = 32;
constexpr uint32_t kWarpNumPerBlock = 16;
constexpr uint32_t kThreadNum = kWarpNumPerBlock * kWarpSize;

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-constant-array-index)
// tl in the same pattern
__global__ void gemm_row_major_no_bias_kernel(const float *input,
                                              const float *weight, float *dst,
                                              uint32_t m, uint32_t d,
                                              uint32_t n) {
  __shared__ typename ::cub::WarpReduce<float>::TempStorage
      temp_storage[kWarpNumPerBlock];

  auto cur_m = blockIdx.y;
  auto cur_n = (blockIdx.x * kWarpNumPerBlock) + (threadIdx.x / kWarpSize);
  if (cur_m < m && cur_n < n) {
    const auto *cur_input = input + (static_cast<size_t>(cur_m) * d);
    const auto *cur_weight = weight + (static_cast<size_t>(cur_n) * d);

    float res{};
    for (uint32_t i = threadIdx.x % kWarpSize; i < d; i += kWarpSize) {
      res += cur_input[i] * cur_weight[i];
    }

    res = ::cub::WarpReduce<float>(temp_storage[threadIdx.x / kWarpSize])
              .Sum(res);

    if (threadIdx.x % 32 == 0) {
      dst[(cur_m * n) + cur_n] = res;
    }
  }
}

__global__ void gemm_row_major_lt_no_bias_kernel(
    const float *input, const float *weight, float *dst, uint32_t b, uint32_t d,
    uint32_t out_head, uint32_t out_d, uint32_t q_start, uint32_t q_end) {
  __shared__ typename ::cub::WarpReduce<float>::TempStorage
      temp_storage[kWarpNumPerBlock];

  auto cur_m = blockIdx.y;
  auto cur_n = (blockIdx.x * kWarpNumPerBlock) + (threadIdx.x / kWarpSize);
  auto cur_h = cur_n / out_d;
  auto cur_d = cur_n % out_d;
  if (cur_m < b && cur_h < out_head && cur_d < out_d) {
    const auto *cur_input = input + (static_cast<size_t>(cur_m) * d);
    const auto *cur_weight = weight + (static_cast<size_t>(cur_n) * d);

    float res{};
    for (uint32_t i = threadIdx.x % kWarpSize; i < d; i += kWarpSize) {
      res += cur_input[i] * cur_weight[i];
    }

    res = ::cub::WarpReduce<float>(temp_storage[threadIdx.x / kWarpSize])
              .Sum(res);

    if (threadIdx.x % 32 == 0) {
      dst[(((((cur_m * out_head) + cur_h) * q_end) + q_start) * out_d) +
          cur_d] = res;
    }
  }
}

__global__ void
gemm_row_major_sl_no_bias_kernel(const float *input, const float *weight,
                                 float *dst, uint32_t b, uint32_t d, uint32_t n,
                                 uint32_t q_start, uint32_t q_end) {
  __shared__ typename ::cub::WarpReduce<float>::TempStorage
      temp_storage[kWarpNumPerBlock];

  auto cur_m = blockIdx.y;
  auto cur_n = (blockIdx.x * kWarpNumPerBlock) + (threadIdx.x / kWarpSize);
  if (cur_m < b && cur_n < n) {
    const auto *cur_input =
        input + (((static_cast<size_t>(cur_m) * q_end) + q_start) * d);
    const auto *cur_weight = weight + (static_cast<size_t>(cur_n) * d);

    float res{};
    for (uint32_t i = threadIdx.x % kWarpSize; i < d; i += kWarpSize) {
      res += cur_input[i] * cur_weight[i];
    }

    res = ::cub::WarpReduce<float>(temp_storage[threadIdx.x / kWarpSize])
              .Sum(res);

    if (threadIdx.x % 32 == 0) {
      dst[(cur_m * n) + cur_n] = res;
    }
  }
}
// NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index)
} // namespace

void gemm_row_major(const float *input, const float *weight, const float *bias,
                    float *dst, uint32_t d, uint32_t n) {
  TINY_LLM_CHECK(tiny_llm::InvalidArgumentError, bias == nullptr);
  if (d == 0 || n == 0) {
    return;
  }
  gemm_row_major_no_bias_kernel<<<CalBlockNum(n, kWarpNumPerBlock), kThreadNum,
                                  0, ThreadCudaContexts::GetContext().stream>>>(
      input, weight, dst, 1, d, n);
}

void gemm_row_major_lt(const float *input, const float *weight,
                       const float *bias, float *dst, uint32_t b, uint32_t d,
                       uint32_t out_head, uint32_t out_d, uint32_t q_start,
                       uint32_t q_end) {
  TINY_LLM_CHECK(tiny_llm::InvalidArgumentError, bias == nullptr);
  TINY_LLM_CHECK(tiny_llm::InvalidArgumentError, q_start + 1 <= q_end);
  if (b == 0 || d == 0 || out_head == 0 || out_d == 0) {
    return;
  }

  gemm_row_major_lt_no_bias_kernel<<<
      dim3{CalBlockNum((out_head * out_d), kWarpNumPerBlock), b}, kThreadNum, 0,
      ThreadCudaContexts::GetContext().stream>>>(
      input, weight, dst, b, d, out_head, out_d, q_start, q_end);
}

void gemm_row_major_tl(const float *input, const float *weight,
                       const float *bias, float *dst, uint32_t b, uint32_t h,
                       uint32_t d, uint32_t out_d) {
  TINY_LLM_CHECK(tiny_llm::InvalidArgumentError, bias == nullptr);
  if (b == 0 || h == 0 || d == 0 || out_d == 0) {
    return;
  }

  gemm_row_major_no_bias_kernel<<<dim3{CalBlockNum(out_d, kWarpNumPerBlock), b},
                                  kThreadNum, 0,
                                  ThreadCudaContexts::GetContext().stream>>>(
      input, weight, dst, b, h * d, out_d);
}

void gemm_row_major_sl(const float *input, const float *weight,
                       const float *bias, float *dst, uint32_t b, uint32_t d,
                       uint32_t n, uint32_t q_start, uint32_t q_end) {
  TINY_LLM_CHECK(tiny_llm::InvalidArgumentError, bias == nullptr);
  TINY_LLM_CHECK(tiny_llm::InvalidArgumentError, q_start + 1 <= q_end);
  if (b == 0 || d == 0 || n == 0) {
    return;
  }

  gemm_row_major_sl_no_bias_kernel<<<dim3{CalBlockNum(n, kWarpNumPerBlock), b},
                                     kThreadNum, 0,
                                     ThreadCudaContexts::GetContext().stream>>>(
      input, weight, dst, b, d, n, q_start, q_end);
}
} // namespace tiny_llm::cuda::q1
