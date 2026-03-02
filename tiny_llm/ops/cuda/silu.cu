#include "tiny_llm/ops/cuda/silu.hpp"

namespace tiny_llm::cuda {
namespace {
constexpr uint32_t kThreadNum = 512;

__global__ void silu_kernel(const float *src, float *dst, size_t element_size) {
  auto index = (static_cast<size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (index < element_size) {
    dst[index] = src[index] / (1 + ::expf(-src[index]));
  }
}
} // namespace

void silu(const float *src, float *dst, size_t element_size) {
  if (element_size == 0) {
    return;
  }
  silu_kernel<<<CalBlockNum(element_size, kThreadNum), kThreadNum, 0,
                ThreadCudaContexts::GetContext().stream>>>(src, dst,
                                                           element_size);
}
} // namespace tiny_llm::cuda
