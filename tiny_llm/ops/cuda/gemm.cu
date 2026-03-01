#include <cstddef>

#include "tiny_llm/common/log_and_excepts.hpp"
#include "tiny_llm/ops/cuda/gemm.hpp"

namespace tiny_llm::cuda {
// NOLINTBEGIN(bugprone-easily-swappable-parameters,cppcoreguidelines-pro-bounds-constant-array-index)
namespace {
constexpr uint32_t kBlockSize = 16;
constexpr uint32_t kTileX = 8;
constexpr uint32_t kTileY = 8;

__global__ void gemm_row_major_plain_no_bias_kernel(const float *input,
                                                    const float *weight,
                                                    float *dst, uint32_t m,
                                                    uint32_t d, uint32_t n) {
  auto cur_x = (blockIdx.x * blockDim.x) + threadIdx.x;
  auto cur_y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (cur_y < m && cur_x < n) {
    float res{};
    const auto *base_input_ptr = input + static_cast<size_t>(cur_y * d);
    const auto *base_weight_ptr = weight + static_cast<size_t>(cur_x * d);
    for (uint32_t i = 0; i < d; ++i) {
      res += base_input_ptr[i] * base_weight_ptr[i];
    }

    dst[(cur_y * n) + cur_x] = res;
  }
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
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
  auto base_x = blockIdx.x * blockDim.x * kTileX;
  auto base_y = blockIdx.y * blockDim.y * kTileY;

  // previous blocks
  for (uint32_t i = 0; i < block_num; ++i) {
    auto buffer_col = (i * kBlockSize) + threadIdx.x;

    for (uint32_t ty = 0; ty < kTileY; ++ty) {
      input_buffer[threadIdx.y][(ty * kBlockSize) + threadIdx.x] = 0;
      auto input_row = base_y + (threadIdx.y * kTileY) + ty;
      if (buffer_col < d && input_row < m) {
        input_buffer[threadIdx.y][(ty * kBlockSize) + threadIdx.x] =
            input[(input_row * d) + buffer_col];
      }
    }
    for (uint32_t tx = 0; tx < kTileX; ++tx) {
      weight_buffer[threadIdx.y][(tx * kBlockSize) + threadIdx.x] = 0;
      auto weight_row = base_x + (threadIdx.y * kTileX) + tx;
      if (buffer_col < d && weight_row < n) {
        weight_buffer[threadIdx.y][(tx * kBlockSize) + threadIdx.x] =
            weight[(weight_row * d) + buffer_col];
      }
    }
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

  for (uint32_t ty = 0; ty < kTileY; ++ty) {
    auto dst_row = base_y + (threadIdx.y * kTileY) + ty;
    for (uint32_t tx = 0; tx < kTileX; ++tx) {
      auto dst_col = base_x + (threadIdx.x * kTileX) + tx;
      if (dst_row < m && dst_col < n) {
        dst[(dst_row * n) + dst_col] = res[ty][tx];
      }
    }
  }
}
} // namespace

void gemm_row_major_plain(const float *input, const float *weight,
                          const float *bias, float *dst, uint32_t m, uint32_t d,
                          uint32_t n) {
  TINY_LLM_CHECK(bias == nullptr);
  if (m * d * n == 0) {
    return;
  }

  gemm_row_major_plain_no_bias_kernel<<<
      dim3{CalBlockNum(n, kBlockSize), CalBlockNum(m, kBlockSize)},
      dim3{kBlockSize, kBlockSize}, 0,
      ThreadCudaContexts::GetContext().stream>>>(input, weight, dst, m, d, n);
}

void gemm_row_major(const float *input, const float *weight, const float *bias,
                    float *dst, uint32_t m, uint32_t d, uint32_t n) {
  TINY_LLM_CHECK(bias == nullptr);
  if (m * d * n == 0) {
    return;
  }

  gemm_row_major_no_bias_kernel<<<dim3{CalBlockNum(n, kBlockSize * kTileX),
                                       CalBlockNum(m, kBlockSize * kTileY)},
                                  dim3{kBlockSize, kBlockSize}, 0,
                                  ThreadCudaContexts::GetContext().stream>>>(
      input, weight, dst, m, d, n);
}
// NOLINTEND(bugprone-easily-swappable-parameters,cppcoreguidelines-pro-bounds-constant-array-index)
} // namespace tiny_llm::cuda
