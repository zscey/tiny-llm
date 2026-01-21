#include "cuda_op_common.hpp"
#include "tiny_llm/ops/cuda/add.hpp"

namespace tiny_llm::cuda {
namespace {
__global__ void add_kernel(const float *a, const float *b, float *dst,
                           size_t size) {
  auto index = (blockIdx.x * ThreadNum1d) + threadIdx.x;
  if (index < size) {
    dst[index] = a[index] + b[index];
  }
}
} // namespace

void add(const float *a, const float *b, float *dst, size_t size,
         cudaStream_t stream) {
  if (size == 0) {
    return;
  }
  add_kernel<<<CalBlockNum(size, ThreadNum1d), ThreadNum1d, 0, stream>>>(
      a, b, dst, size);
}
} // namespace tiny_llm::cuda
