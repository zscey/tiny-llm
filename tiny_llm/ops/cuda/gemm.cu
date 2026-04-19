#include "cccl/cuda/warp"
#include "cub/cub.cuh"
#include "tiny_llm/common/log_and_excepts.hpp"
#include "tiny_llm/ops/cuda/gemm.hpp"
#include <cstddef>

namespace tiny_llm::cuda {
// NOLINTBEGIN(bugprone-easily-swappable-parameters,cppcoreguidelines-pro-bounds-constant-array-index)
namespace {
constexpr uint32_t kBlockSize = 16;
constexpr uint32_t kTileX = 8;
constexpr uint32_t kTileY = 8;

__device__ void fetch_input(const float *input, float *input_buffer,
                            uint32_t col_block_idx, uint32_t m, uint32_t d) {
  auto buffer_col = (col_block_idx * kBlockSize) + threadIdx.x;
  for (uint32_t ty = 0; ty < kTileY; ++ty) {
    auto *buffer_data =
        input_buffer + ((threadIdx.y * ((kBlockSize * kTileY) + 1)) +
                        (ty * kBlockSize) + threadIdx.x);
    *buffer_data = 0.F;
    auto input_row =
        (blockIdx.y * kBlockSize * kTileY) + (ty * kBlockSize) + threadIdx.y;
    if (buffer_col < d && input_row < m) {
      *buffer_data = input[(input_row * d) + buffer_col];
    }
  }
}

__device__ void fetch_weight(const float *weight, float *weight_buffer,
                             uint32_t col_block_idx, uint32_t n, uint32_t d) {
  auto buffer_col = (col_block_idx * kBlockSize) + threadIdx.x;
  for (uint32_t tx = 0; tx < kTileX; ++tx) {
    auto *buffer_data =
        weight_buffer + ((threadIdx.y * ((kBlockSize * kTileX) + 1)) +
                         (tx * kBlockSize) + threadIdx.x);
    *buffer_data = 0.F;
    auto weight_row =
        (blockIdx.x * kBlockSize * kTileX) + (tx * kBlockSize) + threadIdx.y;
    if (buffer_col < d && weight_row < n) {
      *buffer_data = weight[(weight_row * d) + buffer_col];
    }
  }
}

__device__ void write_dst(const float *res, float *dst, uint32_t m,
                          uint32_t n) {
  for (uint32_t ty = 0; ty < kTileY; ++ty) {
    auto dst_row =
        (blockIdx.y * kBlockSize * kTileY) + (ty * kBlockSize) + threadIdx.y;
    for (uint32_t tx = 0; tx < kTileX; ++tx) {
      auto dst_col =
          (blockIdx.x * kBlockSize * kTileX) + (tx * kBlockSize) + threadIdx.x;
      if (dst_row < m && dst_col < n) {
        dst[(dst_row * n) + dst_col] = res[(ty * kTileX) + tx];
      }
    }
  }
}

__global__ void gemm_row_major_no_bias_kernel(const float *input,
                                              const float *weight, float *dst,
                                              uint32_t m, uint32_t d,
                                              uint32_t n) {
  __shared__ float input_buffer[kBlockSize][(kBlockSize * kTileY) + 1];
  __shared__ float weight_buffer[kBlockSize][(kBlockSize * kTileX) + 1];
  float res[kTileY][kTileX];
  float input_regs[kTileY];
  float weight_regs[kTileX];
  for (auto &re : res) {
    for (float &r : re) {
      r = 0.F;
    }
  }

  auto block_num = CalBlockNum(d, kBlockSize);
  for (uint32_t i = 0; i < block_num; ++i) {
    fetch_input(input, &input_buffer[0][0], i, m, d);
    fetch_weight(weight, &weight_buffer[0][0], i, n, d);
    __syncthreads();

    for (uint32_t k = 0; k < kBlockSize; ++k) {
      for (uint32_t ty = 0; ty < kTileY; ++ty) {
        input_regs[ty] = input_buffer[threadIdx.y][(ty * kBlockSize) + k];
      }
      for (uint32_t tx = 0; tx < kTileX; ++tx) {
        weight_regs[tx] = weight_buffer[threadIdx.x][(tx * kBlockSize) + k];
      }

      for (uint32_t ty = 0; ty < kTileY; ++ty) {
        for (uint32_t tx = 0; tx < kTileX; ++tx) {
          res[ty][tx] += input_regs[ty] * weight_regs[tx];
        }
      }
    }
    __syncthreads();
  }

  write_dst(&res[0][0], dst, m, n);
}

__device__ void write_dst_lt(const float *res, float *dst, uint32_t b,
                             uint32_t q, uint32_t out_head, uint32_t out_d,
                             uint32_t q_start, uint32_t q_end) {
  for (uint32_t ty = 0; ty < kTileY; ++ty) {
    auto dst_row =
        (blockIdx.y * kBlockSize * kTileY) + (ty * kBlockSize) + threadIdx.y;
    auto cur_b = dst_row / q;
    auto cur_q = q_start + (dst_row % q);
    for (uint32_t tx = 0; tx < kTileX; ++tx) {
      auto dst_col =
          (blockIdx.x * kBlockSize * kTileX) + (tx * kBlockSize) + threadIdx.x;
      auto cur_h = dst_col / out_d;
      auto cur_d = dst_col % out_d;
      if (cur_b < b && cur_q < q_end && cur_h < out_head && cur_d < out_d) {
        dst[(((((cur_b * out_head) + cur_h) * q_end) + cur_q) * out_d) +
            cur_d] = res[(ty * kTileX) + tx];
      }
    }
  }
}

__global__ void
gemm_row_major_lt_no_bias_kernel(const float *input, const float *weight,
                                 float *dst, uint32_t b, uint32_t q, uint32_t d,
                                 uint32_t out_head, uint32_t out_d,
                                 uint32_t q_start, uint32_t q_end) {
  __shared__ float input_buffer[kBlockSize][(kBlockSize * kTileY) + 1];
  __shared__ float weight_buffer[kBlockSize][(kBlockSize * kTileX) + 1];
  float res[kTileY][kTileX];
  float input_regs[kTileY];
  float weight_regs[kTileX];
  for (auto &re : res) {
    for (float &r : re) {
      r = 0.F;
    }
  }

  auto block_num = CalBlockNum(d, kBlockSize);
  for (uint32_t i = 0; i < block_num; ++i) {
    fetch_input(input, &input_buffer[0][0], i, (b * q), d);
    fetch_weight(weight, &weight_buffer[0][0], i, (out_head * out_d), d);
    __syncthreads();

    for (uint32_t k = 0; k < kBlockSize; ++k) {
      for (uint32_t ty = 0; ty < kTileY; ++ty) {
        input_regs[ty] = input_buffer[threadIdx.y][(ty * kBlockSize) + k];
      }
      for (uint32_t tx = 0; tx < kTileX; ++tx) {
        weight_regs[tx] = weight_buffer[threadIdx.x][(tx * kBlockSize) + k];
      }

      for (uint32_t ty = 0; ty < kTileY; ++ty) {
        for (uint32_t tx = 0; tx < kTileX; ++tx) {
          res[ty][tx] += input_regs[ty] * weight_regs[tx];
        }
      }
    }
    __syncthreads();
  }

  write_dst_lt(&res[0][0], dst, b, q, out_head, out_d, q_start, q_end);
}

__device__ void fetch_input_tl(const float *input, float *input_buffer,
                               uint32_t col_block_idx, uint32_t b, uint32_t h,
                               uint32_t q, uint32_t d) {
  auto buffer_col = (col_block_idx * kBlockSize) + threadIdx.x;
  for (uint32_t ty = 0; ty < kTileY; ++ty) {
    auto *buffer_data =
        input_buffer + ((threadIdx.y * ((kBlockSize * kTileY) + 1)) +
                        (ty * kBlockSize) + threadIdx.x);
    *buffer_data = 0.F;
    auto input_row =
        (blockIdx.y * kBlockSize * kTileY) + (ty * kBlockSize) + threadIdx.y;
    auto cur_b = input_row / q;
    auto cur_q = input_row % q;
    auto cur_h = buffer_col / d;
    auto cur_d = buffer_col % d;
    if (cur_b < b && cur_h < h && cur_q < q && cur_d < d) {
      *buffer_data = input[((((cur_b * h) + cur_h) * q + cur_q) * d) + cur_d];
    }
  }
}

__global__ void gemm_row_major_tl_no_bias_kernel(const float *input,
                                                 const float *weight,
                                                 float *dst, uint32_t b,
                                                 uint32_t h, uint32_t q,
                                                 uint32_t d, uint32_t out_d) {
  __shared__ float input_buffer[kBlockSize][(kBlockSize * kTileY) + 1];
  __shared__ float weight_buffer[kBlockSize][(kBlockSize * kTileX) + 1];
  float res[kTileY][kTileX];
  float input_regs[kTileY];
  float weight_regs[kTileX];
  for (auto &re : res) {
    for (float &r : re) {
      r = 0.F;
    }
  }

  auto block_num = CalBlockNum(h * d, kBlockSize);
  for (uint32_t i = 0; i < block_num; ++i) {
    fetch_input_tl(input, &input_buffer[0][0], i, b, h, q, d);
    fetch_weight(weight, &weight_buffer[0][0], i, out_d, h * d);
    __syncthreads();

    for (uint32_t k = 0; k < kBlockSize; ++k) {
      for (uint32_t ty = 0; ty < kTileY; ++ty) {
        input_regs[ty] = input_buffer[threadIdx.y][(ty * kBlockSize) + k];
      }
      for (uint32_t tx = 0; tx < kTileX; ++tx) {
        weight_regs[tx] = weight_buffer[threadIdx.x][(tx * kBlockSize) + k];
      }

      for (uint32_t ty = 0; ty < kTileY; ++ty) {
        for (uint32_t tx = 0; tx < kTileX; ++tx) {
          res[ty][tx] += input_regs[ty] * weight_regs[tx];
        }
      }
    }
    __syncthreads();
  }

  write_dst(&res[0][0], dst, b * q, out_d);
}

__device__ void fetch_input_sl(const float *input, float *input_buffer,
                               uint32_t col_block_idx, uint32_t b, uint32_t q,
                               uint32_t d, uint32_t q_start, uint32_t q_end) {
  auto buffer_col = (col_block_idx * kBlockSize) + threadIdx.x;
  for (uint32_t ty = 0; ty < kTileY; ++ty) {
    auto *buffer_data =
        input_buffer + ((threadIdx.y * ((kBlockSize * kTileY) + 1)) +
                        (ty * kBlockSize) + threadIdx.x);
    *buffer_data = 0.F;
    auto input_row =
        (blockIdx.y * kBlockSize * kTileY) + (ty * kBlockSize) + threadIdx.y;
    auto cur_b = input_row / q;
    auto cur_q = q_start + (input_row % q);
    if (cur_b < b && cur_q < q_end && buffer_col < d) {
      *buffer_data = input[(((cur_b * q_end) + cur_q) * d) + buffer_col];
    }
  }
}

__global__ void
gemm_row_major_sl_no_bias_kernel(const float *input, const float *weight,
                                 float *dst, uint32_t b, uint32_t q, uint32_t d,
                                 uint32_t n, uint32_t q_start, uint32_t q_end) {
  __shared__ float input_buffer[kBlockSize][(kBlockSize * kTileY) + 1];
  __shared__ float weight_buffer[kBlockSize][(kBlockSize * kTileX) + 1];
  float res[kTileY][kTileX];
  float input_regs[kTileY];
  float weight_regs[kTileX];
  for (auto &re : res) {
    for (float &r : re) {
      r = 0.F;
    }
  }

  auto block_num = CalBlockNum(d, kBlockSize);
  for (uint32_t i = 0; i < block_num; ++i) {
    fetch_input_sl(input, &input_buffer[0][0], i, b, q, d, q_start, q_end);
    fetch_weight(weight, &weight_buffer[0][0], i, n, d);
    __syncthreads();

    for (uint32_t k = 0; k < kBlockSize; ++k) {
      for (uint32_t ty = 0; ty < kTileY; ++ty) {
        input_regs[ty] = input_buffer[threadIdx.y][(ty * kBlockSize) + k];
      }
      for (uint32_t tx = 0; tx < kTileX; ++tx) {
        weight_regs[tx] = weight_buffer[threadIdx.x][(tx * kBlockSize) + k];
      }

      for (uint32_t ty = 0; ty < kTileY; ++ty) {
        for (uint32_t tx = 0; tx < kTileX; ++tx) {
          res[ty][tx] += input_regs[ty] * weight_regs[tx];
        }
      }
    }
    __syncthreads();
  }

  write_dst(&res[0][0], dst, b * q, n);
}

namespace ty1 {
constexpr uint32_t kWarpSize = 32;
constexpr uint32_t kWarpNumPerBlock = 16;
constexpr uint32_t kThreadNum = kWarpNumPerBlock * kWarpSize;

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

// q1
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

// q1
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
} // namespace ty1
} // namespace

void gemm_row_major(const float *input, const float *weight, const float *bias,
                    float *dst, uint32_t m, uint32_t d, uint32_t n) {
  TINY_LLM_CHECK(bias == nullptr);
  if (m == 0 || d == 0 || n == 0) {
    return;
  }
  if (m == 1) {
    ty1::gemm_row_major_no_bias_kernel<<<
        CalBlockNum(n, ty1::kWarpNumPerBlock), ty1::kThreadNum, 0,
        ThreadCudaContexts::GetContext().stream>>>(input, weight, dst, m, d, n);
    return;
  }

  gemm_row_major_no_bias_kernel<<<dim3{CalBlockNum(n, kBlockSize * kTileX),
                                       CalBlockNum(m, kBlockSize * kTileY)},
                                  dim3{kBlockSize, kBlockSize}, 0,
                                  ThreadCudaContexts::GetContext().stream>>>(
      input, weight, dst, m, d, n);
}

void gemm_row_major_lt(const float *input, const float *weight,
                       const float *bias, float *dst, uint32_t b, uint32_t q,
                       uint32_t d, uint32_t out_head, uint32_t out_d,
                       uint32_t q_start, uint32_t q_end) {
  TINY_LLM_CHECK(bias == nullptr);
  TINY_LLM_CHECK(q_start + q <= q_end);
  if (b == 0 || q == 0 || d == 0 || out_head == 0 || out_d == 0) {
    return;
  }

  if (q == 1) {
    ty1::gemm_row_major_lt_no_bias_kernel<<<
        dim3{CalBlockNum((out_head * out_d), ty1::kWarpNumPerBlock), b},
        ty1::kThreadNum, 0, ThreadCudaContexts::GetContext().stream>>>(
        input, weight, dst, b, d, out_head, out_d, q_start, q_end);
    return;
  }

  gemm_row_major_lt_no_bias_kernel<<<
      dim3{CalBlockNum((out_head * out_d), kBlockSize * kTileX),
           CalBlockNum((b * q), kBlockSize * kTileY)},
      dim3{kBlockSize, kBlockSize}, 0,
      ThreadCudaContexts::GetContext().stream>>>(
      input, weight, dst, b, q, d, out_head, out_d, q_start, q_end);
}

void gemm_row_major_tl(const float *input, const float *weight,
                       const float *bias, float *dst, uint32_t b, uint32_t h,
                       uint32_t q, uint32_t d, uint32_t out_d) {
  TINY_LLM_CHECK(bias == nullptr);
  if (b == 0 || h == 0 || q == 0 || d == 0 || out_d == 0) {
    return;
  }

  if (q == 1) {
    ty1::gemm_row_major_no_bias_kernel<<<
        dim3{CalBlockNum(out_d, ty1::kWarpNumPerBlock), b}, ty1::kThreadNum, 0,
        ThreadCudaContexts::GetContext().stream>>>(input, weight, dst, b, h * d,
                                                   out_d);
    return;
  }

  gemm_row_major_tl_no_bias_kernel<<<
      dim3{CalBlockNum(out_d, kBlockSize * kTileX),
           CalBlockNum((b * q), kBlockSize * kTileY)},
      dim3{kBlockSize, kBlockSize}, 0,
      ThreadCudaContexts::GetContext().stream>>>(input, weight, dst, b, h, q, d,
                                                 out_d);
}

void gemm_row_major_sl(const float *input, const float *weight,
                       const float *bias, float *dst, uint32_t b, uint32_t q,
                       uint32_t d, uint32_t n, uint32_t q_start,
                       uint32_t q_end) {
  TINY_LLM_CHECK(bias == nullptr);
  TINY_LLM_CHECK(q_start + q <= q_end);
  if (b == 0 || q == 0 || d == 0 || n == 0) {
    return;
  }

  if (q == 1) {
    ty1::gemm_row_major_sl_no_bias_kernel<<<
        dim3{CalBlockNum(n, ty1::kWarpNumPerBlock), b}, ty1::kThreadNum, 0,
        ThreadCudaContexts::GetContext().stream>>>(input, weight, dst, b, d, n,
                                                   q_start, q_end);
    return;
  }

  gemm_row_major_sl_no_bias_kernel<<<dim3{CalBlockNum(n, kBlockSize * kTileX),
                                          CalBlockNum((b * q),
                                                      kBlockSize * kTileY)},
                                     dim3{kBlockSize, kBlockSize}, 0,
                                     ThreadCudaContexts::GetContext().stream>>>(
      input, weight, dst, b, q, d, n, q_start, q_end);
}
// NOLINTEND(bugprone-easily-swappable-parameters,cppcoreguidelines-pro-bounds-constant-array-index)
} // namespace tiny_llm::cuda
