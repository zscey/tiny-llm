#include "tiny_llm/ops/cuda/arithmetic.hpp"

namespace tiny_llm::cuda {
namespace {
constexpr uint32_t kThreadNum = 512;

template <ArithmeticType T>
__global__ void arithmetic_kernel(const float *left, const float *right,
                                  float *dst, size_t element_size) {
  auto cur_idx = (static_cast<size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (cur_idx < element_size) {
    if constexpr (T == ArithmeticType::kAdd) {
      dst[cur_idx] = left[cur_idx] + right[cur_idx];
    } else if constexpr (T == ArithmeticType::kMul) {
      dst[cur_idx] = left[cur_idx] * right[cur_idx];
    } else if constexpr (T == ArithmeticType::kSub) {
      dst[cur_idx] = left[cur_idx] - right[cur_idx];
    } else if constexpr (T == ArithmeticType::kDiv) {
      dst[cur_idx] = left[cur_idx] / right[cur_idx];
    }
  }
}
} // namespace

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define CASE_OF_TYPE(type)                                                     \
  case type:                                                                   \
    arithmetic_kernel<type>                                                    \
        <<<CalBlockNum(element_size, kThreadNum), kThreadNum, 0,               \
           context.stream>>>(left, right, dst, element_size);                  \
    return;

void arithmetic(const float *left, const float *right, float *dst,
                size_t element_size, ArithmeticType type) {
  if (element_size == 0) {
    return;
  }

  auto context = ThreadCudaContexts::GetContext();
  switch (type) {
    CASE_OF_TYPE(ArithmeticType::kAdd);
    CASE_OF_TYPE(ArithmeticType::kMul);
    CASE_OF_TYPE(ArithmeticType::kSub);
    CASE_OF_TYPE(ArithmeticType::kDiv);
  }
}
} // namespace tiny_llm::cuda
