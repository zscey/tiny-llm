#include "cuda_op_common.hpp"
#include "tiny_llm/common/log_and_excepts.hpp"
#include "tiny_llm/ops/cuda/rope.hpp"

namespace tiny_llm::cuda {
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
namespace {
constexpr uint32_t kThreadNum = 512;

__global__ void rope_kernel(float *cos_dst, float *sin_dst, uint32_t max_len,
                            uint32_t half_dim, double base) {
  auto t_shift = (static_cast<size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  auto seq_idx = t_shift / half_dim;
  if (seq_idx < max_len) {
    auto dim_idx = static_cast<double>(t_shift % half_dim);
    auto scaled_theta = static_cast<double>(seq_idx) *
                        ::pow(base, -dim_idx / static_cast<double>(half_dim));

    auto shift = t_shift + (seq_idx * half_dim);
    auto cos_value = static_cast<float>(::cos(scaled_theta));
    cos_dst[shift] = cos_value;
    cos_dst[shift + half_dim] = cos_value;
    auto sin_value = static_cast<float>(::sin(scaled_theta));
    sin_dst[shift] = sin_value;
    sin_dst[shift + half_dim] = sin_value;
  }
}
} // namespace

void rope(float *cos_dst, float *sin_dst, uint32_t max_len, uint32_t dim,
          double base) {
  if (max_len == 0 || dim == 0) {
    return;
  }
  TINY_LLM_CHECK((dim & 1U) == 0);

  auto half_dim = dim / 2;
  rope_kernel<<<CalBlockNum(max_len * half_dim, kThreadNum), kThreadNum, 0,
                ThreadCudaContexts::GetContext().stream>>>(
      cos_dst, sin_dst, max_len, half_dim, base);
}
// NOLINTEND(bugprone-easily-swappable-parameters)
} // namespace tiny_llm::cuda
