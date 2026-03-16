
#include "cccl/cuda/warp"
#include "cub/cub.cuh"
#include "cuda_op_common.hpp"
#include "tiny_llm/common/log_and_excepts.hpp"
#include "tiny_llm/ops/cuda/norm.hpp"

namespace tiny_llm::cuda {
// NOLINTBEGIN(bugprone-easily-swappable-parameters,cppcoreguidelines-pro-bounds-constant-array-index)
namespace {
constexpr uint32_t kWarpSize = 32;
constexpr uint32_t kWarpNumPerBlock = 16;
constexpr uint32_t kWarpIterPerBlock = 4;
constexpr uint32_t kBlockSize = kWarpIterPerBlock * kWarpNumPerBlock;
constexpr uint32_t kThreadNum = kWarpNumPerBlock * kWarpSize;

template <uint32_t OutPerThread>
__global__ void rms_norm_kernel(const float *input, const float *weight,
                                float *dst, size_t element_size, uint32_t dim,
                                float eps) {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  extern __shared__ float buffer[];
  for (size_t cur_idx = threadIdx.x; cur_idx < dim; cur_idx += blockDim.x) {
    buffer[cur_idx] = weight[cur_idx];
  }
  __syncthreads();

  __shared__ typename ::cub::WarpReduce<float>::TempStorage
      temp_storage[kWarpNumPerBlock];
  float scaled_input[OutPerThread];
  for (size_t iter = 0; iter < kWarpIterPerBlock; ++iter) {
    auto cur_row = (static_cast<size_t>(blockIdx.x) * kBlockSize) +
                   (iter * kWarpNumPerBlock) + (threadIdx.x / kWarpSize);
    if (cur_row >= element_size) {
      continue;
    }

    const auto *cur_input = input + (cur_row * dim);
    float rms{};
    for (uint32_t i = 0, cur_col = threadIdx.x % kWarpSize; cur_col < dim;
         ++i, cur_col += kWarpSize) {
      auto value = cur_input[cur_col];
      scaled_input[i] = buffer[cur_col] * value;
      rms += ::pow(value, 2) / static_cast<float>(dim);
    }

    rms = ::sqrt(::cub::WarpReduce<float>(temp_storage[threadIdx.x / kWarpSize])
                     .Sum(rms) +
                 eps);
    rms = ::cuda::device::warp_shuffle_idx(rms, 0);

    auto *cur_dst = dst + (cur_row * dim);
    for (uint32_t i = 0, cur_col = threadIdx.x % kWarpSize; cur_col < dim;
         ++i, cur_col += kWarpSize) {
      cur_dst[cur_col] = scaled_input[i] / rms;
    }
  }
}
} // namespace

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define CALL_KERNEL(OutPerThread)                                              \
  rms_norm_kernel<OutPerThread>                                                \
      <<<CalBlockNum(element_size, kBlockSize), kThreadNum,                    \
         sizeof(float) * dim, ThreadCudaContexts::GetContext().stream>>>(      \
          input, weight, dst, element_size, dim, eps);

void rms_norm(const float *input, const float *weight, float *dst,
              size_t element_size, uint32_t dim, float eps) {
  if (element_size == 0) {
    return;
  }
  TINY_LLM_CHECK(dim > 0);

  if (dim <= 64) {
    CALL_KERNEL(2);
    return;
  }
  if (dim <= 128) {
    CALL_KERNEL(4);
    return;
  }
  if (dim <= 256) {
    CALL_KERNEL(8);
    return;
  }

  TINY_LLM_THROW_ERROR(std::runtime_error,
                       "The condition `dim={}` is not implemented.", dim);
}
// NOLINTEND(bugprone-easily-swappable-parameters,cppcoreguidelines-pro-bounds-constant-array-index)
} // namespace tiny_llm::cuda
