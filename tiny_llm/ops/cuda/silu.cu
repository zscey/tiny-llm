#include "tiny_llm/ops/cuda/silu.hpp"

namespace tiny_llm::cuda {
namespace {
__global__ void silu_kernel(const float *src, float *dst, size_t size) {
  auto index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < size) {
    dst[index] = src[index] / (1 + ::expf(-src[index]));
  }
}
} // namespace

void silu(const float *src, float *dst, size_t size) {
  if (size == 0) {
    return;
  }
  silu_kernel<<<CalBlockNum(size, ThreadNum1d), ThreadNum1d, 0,
                ThreadCudaContexts::GetContext().stream>>>(src, dst, size);
}
} // namespace tiny_llm::cuda
