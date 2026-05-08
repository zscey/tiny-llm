#include "tiny_llm/common/checks.hpp"
#include "tiny_llm/common/exception.hpp"
#include "tiny_llm/ops/cuda/gemm.hpp"
#include "tiny_llm/ops/cuda/gemm_q1_helper.hpp"
#include <cstddef>

namespace tiny_llm::cuda {
// NOLINTBEGIN(cppcoreguidelines-pro-bounds-constant-array-index)
namespace {
constexpr uint32_t kThreadNumX = 32;
constexpr uint32_t kThreadNumY = 8;
static_assert(kThreadNumX % kThreadNumY == 0);
constexpr uint32_t kTileX = 8;
constexpr uint32_t kTileY = 8;

__device__ void fetch_input(const float *input, float *input_buffer,
                            uint32_t col_block_idx, uint32_t m, uint32_t d) {
  auto buffer_col = (col_block_idx * kThreadNumX) + threadIdx.x;

  for (uint32_t ty = 0; ty < kTileY; ++ty) {
    auto input_row =
        (blockIdx.y * kTileY * kThreadNumY) + (ty * kThreadNumY) + threadIdx.y;
    float val = 0.F;
    if (input_row < m && buffer_col < d) {
      val = input[(input_row * d) + buffer_col];
    }
    input_buffer[(((ty * kThreadNumY) + threadIdx.y) * kThreadNumX) +
                 threadIdx.x] = val;
  }
}

__device__ void fetch_weight(const float *weight, float *weight_buffer,
                             uint32_t col_block_idx, uint32_t n, uint32_t d) {
  auto buffer_col = (col_block_idx * kThreadNumX) + threadIdx.x;

  for (uint32_t tx = 0; tx < kTileX; ++tx) {
    for (uint32_t i = 0; i < kThreadNumX / kThreadNumY; ++i) {
      float val = 0.F;
      auto weight_row = (blockIdx.x * kTileX * kThreadNumX) +
                        (tx * kThreadNumX) + (i * kThreadNumY) + threadIdx.y;
      if (weight_row < n && buffer_col < d) {
        val = weight[(weight_row * d) + buffer_col];
      }
      weight_buffer[(threadIdx.x * ((kTileX * kThreadNumX) + 1)) +
                    (tx * kThreadNumX) + (i * kThreadNumY) + threadIdx.y] = val;
    }
  }
}

__device__ void write_dst(const float *res, float *dst, uint32_t m,
                          uint32_t n) {
  for (uint32_t ty = 0; ty < kTileY; ++ty) {
    auto dst_row =
        (blockIdx.y * kThreadNumY * kTileY) + (ty * kThreadNumY) + threadIdx.y;
    for (uint32_t tx = 0; tx < kTileX; ++tx) {
      auto dst_col = (blockIdx.x * kThreadNumX * kTileX) + (tx * kThreadNumX) +
                     threadIdx.x;
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
  __shared__ float input_buffer[kTileY * kThreadNumY][kThreadNumX];
  __shared__ float weight_buffer[kThreadNumX][(kTileX * kThreadNumX) + 1];
  float res[kTileY][kTileX];
  for (auto &re : res) {
    for (float &r : re) {
      r = 0.F;
    }
  }

  auto block_num = CalBlockNum(d, kThreadNumX);
  for (uint32_t i = 0; i < block_num; ++i) {
    fetch_input(input, &input_buffer[0][0], i, m, d);
    fetch_weight(weight, &weight_buffer[0][0], i, n, d);
    __syncthreads();

    for (uint32_t k = 0; k < kThreadNumX; ++k) {
      float input_regs[kTileY];
      float weight_regs[kTileX];
      for (uint32_t ty = 0; ty < kTileY; ++ty) {
        input_regs[ty] = input_buffer[(ty * kThreadNumY) + threadIdx.y][k];
      }
      for (uint32_t tx = 0; tx < kTileX; ++tx) {
        weight_regs[tx] = weight_buffer[k][(tx * kThreadNumX) + threadIdx.x];
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
        (blockIdx.y * kThreadNumY * kTileY) + (ty * kThreadNumY) + threadIdx.y;
    auto cur_b = dst_row / q;
    auto cur_q = q_start + (dst_row % q);
    for (uint32_t tx = 0; tx < kTileX; ++tx) {
      auto dst_col = (blockIdx.x * kThreadNumX * kTileX) + (tx * kThreadNumX) +
                     threadIdx.x;
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
  __shared__ float input_buffer[kTileY * kThreadNumY][kThreadNumX];
  __shared__ float weight_buffer[kThreadNumX][(kTileX * kThreadNumX) + 1];
  float res[kTileY][kTileX];
  for (auto &re : res) {
    for (float &r : re) {
      r = 0.F;
    }
  }

  auto block_num = CalBlockNum(d, kThreadNumX);
  for (uint32_t i = 0; i < block_num; ++i) {
    fetch_input(input, &input_buffer[0][0], i, (b * q), d);
    fetch_weight(weight, &weight_buffer[0][0], i, (out_head * out_d), d);
    __syncthreads();

    for (uint32_t k = 0; k < kThreadNumX; ++k) {
      float input_regs[kTileY];
      float weight_regs[kTileX];
      for (uint32_t ty = 0; ty < kTileY; ++ty) {
        input_regs[ty] = input_buffer[(ty * kThreadNumY) + threadIdx.y][k];
      }
      for (uint32_t tx = 0; tx < kTileX; ++tx) {
        weight_regs[tx] = weight_buffer[k][(tx * kThreadNumX) + threadIdx.x];
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
  auto buffer_col = (col_block_idx * kThreadNumX) + threadIdx.x;

  for (uint32_t ty = 0; ty < kTileY; ++ty) {
    float val = 0.F;
    auto input_row =
        (blockIdx.y * kThreadNumY * kTileY) + (ty * kThreadNumY) + threadIdx.y;
    auto cur_b = input_row / q;
    auto cur_q = input_row % q;
    auto cur_h = buffer_col / d;
    auto cur_d = buffer_col % d;
    if (cur_b < b && cur_h < h && cur_q < q && cur_d < d) {
      val = input[((((cur_b * h) + cur_h) * q + cur_q) * d) + cur_d];
    }
    input_buffer[(((ty * kThreadNumY) + threadIdx.y) * kThreadNumX) +
                 threadIdx.x] = val;
  }
}

__global__ void gemm_row_major_tl_no_bias_kernel(const float *input,
                                                 const float *weight,
                                                 float *dst, uint32_t b,
                                                 uint32_t h, uint32_t q,
                                                 uint32_t d, uint32_t out_d) {
  __shared__ float input_buffer[kTileY * kThreadNumY][kThreadNumX];
  __shared__ float weight_buffer[kThreadNumX][(kTileX * kThreadNumX) + 1];
  float res[kTileY][kTileX];
  for (auto &re : res) {
    for (float &r : re) {
      r = 0.F;
    }
  }

  auto block_num = CalBlockNum(h * d, kThreadNumX);
  for (uint32_t i = 0; i < block_num; ++i) {
    fetch_input_tl(input, &input_buffer[0][0], i, b, h, q, d);
    fetch_weight(weight, &weight_buffer[0][0], i, out_d, h * d);
    __syncthreads();

    for (uint32_t k = 0; k < kThreadNumX; ++k) {
      float input_regs[kTileY];
      float weight_regs[kTileX];
      for (uint32_t ty = 0; ty < kTileY; ++ty) {
        input_regs[ty] = input_buffer[(ty * kThreadNumY) + threadIdx.y][k];
      }
      for (uint32_t tx = 0; tx < kTileX; ++tx) {
        weight_regs[tx] = weight_buffer[k][(tx * kThreadNumX) + threadIdx.x];
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
  auto buffer_col = (col_block_idx * kThreadNumX) + threadIdx.x;
  for (uint32_t ty = 0; ty < kTileY; ++ty) {
    float val = 0.F;
    auto input_row =
        (blockIdx.y * kThreadNumY * kTileY) + (ty * kThreadNumY) + threadIdx.y;
    auto cur_b = input_row / q;
    auto cur_q = q_start + (input_row % q);
    if (cur_b < b && cur_q < q_end && buffer_col < d) {
      val = input[(((cur_b * q_end) + cur_q) * d) + buffer_col];
    }
    input_buffer[(((ty * kThreadNumY) + threadIdx.y) * kThreadNumX) +
                 threadIdx.x] = val;
  }
}

__global__ void
gemm_row_major_sl_no_bias_kernel(const float *input, const float *weight,
                                 float *dst, uint32_t b, uint32_t q, uint32_t d,
                                 uint32_t n, uint32_t q_start, uint32_t q_end) {
  __shared__ float input_buffer[kTileY * kThreadNumY][kThreadNumX];
  __shared__ float weight_buffer[kThreadNumX][(kTileX * kThreadNumX) + 1];
  float res[kTileY][kTileX];
  for (auto &re : res) {
    for (float &r : re) {
      r = 0.F;
    }
  }

  auto block_num = CalBlockNum(d, kThreadNumX);
  for (uint32_t i = 0; i < block_num; ++i) {
    fetch_input_sl(input, &input_buffer[0][0], i, b, q, d, q_start, q_end);
    fetch_weight(weight, &weight_buffer[0][0], i, n, d);
    __syncthreads();

    for (uint32_t k = 0; k < kThreadNumX; ++k) {
      float input_regs[kTileY];
      float weight_regs[kTileX];
      for (uint32_t ty = 0; ty < kTileY; ++ty) {
        input_regs[ty] = input_buffer[(ty * kThreadNumY) + threadIdx.y][k];
      }
      for (uint32_t tx = 0; tx < kTileX; ++tx) {
        weight_regs[tx] = weight_buffer[k][(tx * kThreadNumX) + threadIdx.x];
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
} // namespace

void gemm_row_major(const float *input, const float *weight, const float *bias,
                    float *dst, uint32_t m, uint32_t d, uint32_t n) {
  if (m == 1) {
    q1::gemm_row_major(input, weight, bias, dst, d, n);
    return;
  }

  TINY_LLM_CHECK(tiny_llm::InvalidArgumentError, bias == nullptr);
  if (m == 0 || d == 0 || n == 0) {
    return;
  }
  gemm_row_major_no_bias_kernel<<<dim3{CalBlockNum(n, kThreadNumX * kTileX),
                                       CalBlockNum(m, kThreadNumY * kTileY)},
                                  dim3{kThreadNumX, kThreadNumY}, 0,
                                  ThreadCudaContexts::GetContext().stream>>>(
      input, weight, dst, m, d, n);
}

void gemm_row_major_lt(const float *input, const float *weight,
                       const float *bias, float *dst, uint32_t b, uint32_t q,
                       uint32_t d, uint32_t out_head, uint32_t out_d,
                       uint32_t q_start, uint32_t q_end) {
  if (q == 1) {
    q1::gemm_row_major_lt(input, weight, bias, dst, b, d, out_head, out_d,
                          q_start, q_end);
    return;
  }

  TINY_LLM_CHECK(tiny_llm::InvalidArgumentError, bias == nullptr);
  TINY_LLM_CHECK(tiny_llm::InvalidArgumentError, q_start + q <= q_end);
  if (b == 0 || q == 0 || d == 0 || out_head == 0 || out_d == 0) {
    return;
  }
  gemm_row_major_lt_no_bias_kernel<<<
      dim3{CalBlockNum((out_head * out_d), kThreadNumX * kTileX),
           CalBlockNum((b * q), kThreadNumY * kTileY)},
      dim3{kThreadNumX, kThreadNumY}, 0,
      ThreadCudaContexts::GetContext().stream>>>(
      input, weight, dst, b, q, d, out_head, out_d, q_start, q_end);
}

void gemm_row_major_tl(const float *input, const float *weight,
                       const float *bias, float *dst, uint32_t b, uint32_t h,
                       uint32_t q, uint32_t d, uint32_t out_d) {
  if (q == 1) {
    q1::gemm_row_major_tl(input, weight, bias, dst, b, h, d, out_d);
    return;
  }

  TINY_LLM_CHECK(tiny_llm::InvalidArgumentError, bias == nullptr);
  if (b == 0 || h == 0 || q == 0 || d == 0 || out_d == 0) {
    return;
  }
  gemm_row_major_tl_no_bias_kernel<<<
      dim3{CalBlockNum(out_d, kThreadNumX * kTileX),
           CalBlockNum((b * q), kThreadNumY * kTileY)},
      dim3{kThreadNumX, kThreadNumY}, 0,
      ThreadCudaContexts::GetContext().stream>>>(input, weight, dst, b, h, q, d,
                                                 out_d);
}

void gemm_row_major_sl(const float *input, const float *weight,
                       const float *bias, float *dst, uint32_t b, uint32_t q,
                       uint32_t d, uint32_t n, uint32_t q_start,
                       uint32_t q_end) {
  if (q == 1) {
    q1::gemm_row_major_sl(input, weight, bias, dst, b, d, n, q_start, q_end);
    return;
  }

  TINY_LLM_CHECK(tiny_llm::InvalidArgumentError, bias == nullptr);
  TINY_LLM_CHECK(tiny_llm::InvalidArgumentError, q_start + q <= q_end);
  if (b == 0 || q == 0 || d == 0 || n == 0) {
    return;
  }
  gemm_row_major_sl_no_bias_kernel<<<dim3{CalBlockNum(n, kThreadNumX * kTileX),
                                          CalBlockNum((b * q),
                                                      kThreadNumY * kTileY)},
                                     dim3{kThreadNumX, kThreadNumY}, 0,
                                     ThreadCudaContexts::GetContext().stream>>>(
      input, weight, dst, b, q, d, n, q_start, q_end);
}
// NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index)
} // namespace tiny_llm::cuda
