#include "cuda_op_common.hpp"
#include "tiny_llm/common/checks.hpp"
#include "tiny_llm/common/exception.hpp"
#include "tiny_llm/ops/cuda/rope.hpp"

namespace tiny_llm::cuda {
namespace {
constexpr uint32_t kThreadNum = 512;

__global__ void rope_kernel(float *cos_dst, float *sin_dst, uint32_t max_len,
                            uint32_t half_dim, double base) {
  auto shift = (static_cast<size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  auto seq_idx = shift / half_dim;
  if (seq_idx < max_len) {
    auto dim_idx = static_cast<double>(shift % half_dim);
    auto scaled_theta = static_cast<double>(seq_idx) *
                        ::pow(base, -dim_idx / static_cast<double>(half_dim));

    cos_dst[shift] = static_cast<float>(::cos(scaled_theta));
    sin_dst[shift] = static_cast<float>(::sin(scaled_theta));
  }
}

__global__ void apply_rope_inplace_kernel(const float *cos, const float *sin,
                                          const uint32_t *position_ids,
                                          float *dst, size_t element_size,
                                          uint32_t head_num, uint32_t seq_len,
                                          uint32_t half_dim, uint32_t seq_start,
                                          uint32_t seq_end) {
  auto t_shift = (static_cast<size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  auto element_idx = t_shift / half_dim;
  if (element_idx < element_size) {
    auto dim_idx = t_shift % half_dim;
    auto pe_shift =
        (static_cast<size_t>(position_ids[element_idx]) * half_dim) + dim_idx;
    auto cos_factor = cos[pe_shift];
    auto sin_factor = sin[pe_shift];

    auto batch_idx = element_idx / seq_len;
    auto seq_idx = (element_idx % seq_len) + seq_start;
    auto *real_ptr =
        dst +
        (((batch_idx * head_num * seq_end + seq_idx) * half_dim * 2) + dim_idx);
    for (uint32_t i = 0; i < head_num; ++i) {
      auto *imaginary_ptr = real_ptr + half_dim;
      auto real = *real_ptr;
      auto imaginary = *imaginary_ptr;
      *real_ptr = (cos_factor * real) - (sin_factor * imaginary);
      *imaginary_ptr = (sin_factor * real) + (cos_factor * imaginary);

      real_ptr += static_cast<size_t>(seq_end) * half_dim * 2;
    }
  }
}
} // namespace

void rope(float *cos_dst, float *sin_dst, uint32_t max_len, uint32_t dim,
          double base) {
  TINY_LLM_CHECK(tiny_llm::InvalidArgumentError, max_len > 0);
  TINY_LLM_CHECK(tiny_llm::InvalidArgumentError, dim > 0);
  TINY_LLM_CHECK(tiny_llm::InvalidArgumentError, (dim & 1U) == 0);

  auto half_dim = dim / 2;
  rope_kernel<<<CalBlockNum(max_len * half_dim, kThreadNum), kThreadNum, 0,
                ThreadCudaContexts::GetContext().stream>>>(
      cos_dst, sin_dst, max_len, half_dim, base);
}

void apply_rope_inplace(const float *cos, const float *sin,
                        const uint32_t *position_ids, float *dst,
                        uint32_t batch, uint32_t head_num, uint32_t seq_len,
                        uint32_t dim, uint32_t seq_start, uint32_t seq_end) {
  auto element_size = static_cast<size_t>(batch) * seq_len;
  if (element_size == 0) {
    return;
  }
  TINY_LLM_CHECK(tiny_llm::InvalidArgumentError, head_num > 0);
  TINY_LLM_CHECK(tiny_llm::InvalidArgumentError, dim > 0);
  TINY_LLM_CHECK(tiny_llm::InvalidArgumentError, (dim & 1U) == 0);
  TINY_LLM_CHECK(tiny_llm::InvalidArgumentError,
                 seq_start + seq_len <= seq_end);

  auto half_dim = dim / 2;
  apply_rope_inplace_kernel<<<CalBlockNum(element_size * half_dim, kThreadNum),
                              kThreadNum, 0,
                              ThreadCudaContexts::GetContext().stream>>>(
      cos, sin, position_ids, dst, element_size, head_num, seq_len, half_dim,
      seq_start, seq_end);
}
} // namespace tiny_llm::cuda
