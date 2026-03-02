#include "cuda_op_common.hpp"
#include "tiny_llm/ops/cuda/embedding.hpp"

namespace tiny_llm::cuda {
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
namespace {
constexpr uint32_t kThreadNum = 512;
constexpr uint32_t kTile = 8;

__global__ void embedding_kernel(const uint32_t *src, const float *emb_weights,
                                 float *dst, size_t padded_dim, size_t dim,
                                 size_t element_size) {
  auto shift =
      ((static_cast<size_t>(blockIdx.x) * blockDim.x) + threadIdx.x) * kTile;
  auto element_idx = shift / padded_dim;
  if (element_idx < element_size) {
    auto dim_idx = shift % padded_dim;
    const auto *emb_weights_ptr =
        emb_weights + (src[element_idx] * dim) + dim_idx;
    const auto *emb_weights_end =
        emb_weights_ptr + ::min(static_cast<size_t>(kTile), dim - dim_idx);
    auto *dst_ptr = dst + (element_idx * dim) + dim_idx;
    while (emb_weights_ptr != emb_weights_end) {
      *dst_ptr = *emb_weights_ptr;
      ++emb_weights_ptr;
      ++dst_ptr;
    }
  }
}
} // namespace

void embedding(const uint32_t *src, const float *emb_weights, float *dst,
               uint32_t dim, size_t element_size) {
  if (element_size == 0 || dim == 0) {
    return;
  }

  auto padded_dim = CalBlockNum(dim, kTile) * kTile;
  embedding_kernel<<<CalBlockNum(element_size * padded_dim,
                                 static_cast<size_t>(kThreadNum * kTile)),
                     kThreadNum, 0, ThreadCudaContexts::GetContext().stream>>>(
      src, emb_weights, dst, padded_dim, dim, element_size);
}
// NOLINTEND(bugprone-easily-swappable-parameters)
} // namespace tiny_llm::cuda
