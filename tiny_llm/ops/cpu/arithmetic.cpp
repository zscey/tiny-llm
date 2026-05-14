#include "tiny_llm/ops/cpu/arithmetic.hpp"
#ifdef TINY_LLM_CPU_OPS_WITH_AVX2
#include <immintrin.h>
#endif

namespace tiny_llm::cpu {
namespace {
namespace avx2 {
#ifdef TINY_LLM_CPU_OPS_WITH_AVX2
auto arithmetic_mul_avx2_kernel(const float *left, float right, float *dst,
                                size_t beg, size_t end) -> size_t {
  auto right_vec = _mm256_set1_ps(right);

  auto vec_num = (end - beg) / 8 * 8;
  for (end = beg + vec_num; beg < end; beg += 8) {
    auto left_vec = _mm256_loadu_ps(left + beg);
    // NOLINTNEXTLINE(portability-simd-intrinsics)
    auto res = _mm256_mul_ps(left_vec, right_vec);
    _mm256_storeu_ps(dst + beg, res);
  }

  return vec_num;
}
#else
constexpr auto arithmetic_mul_avx2_kernel(const float * /*left*/,
                                          float /*right*/, float * /*dst*/,
                                          size_t /*beg*/, size_t /*end*/)
    -> size_t {
  return 0;
}
#endif
} // namespace avx2

template <ArithmeticType T>
void arithmetic_kernel(const float *left, float right, float *dst, size_t beg,
                       size_t end) {
  static_assert(T == ArithmeticType::kAdd || T == ArithmeticType::kMul ||
                T == ArithmeticType::kSub);

  if constexpr (T == ArithmeticType::kAdd) {
    for (; beg < end; ++beg) {
      dst[beg] = left[beg] + right;
    }
  } else if constexpr (T == ArithmeticType::kMul) {
    for (beg += avx2::arithmetic_mul_avx2_kernel(left, right, dst, beg, end);
         beg < end; ++beg) {
      dst[beg] = left[beg] * right;
    }
  } else if constexpr (T == ArithmeticType::kSub) {
    for (; beg < end; ++beg) {
      dst[beg] = left[beg] - right;
    }
  }
}
} // namespace

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define RUN_KERNEL(type)                                                       \
  parallel_run(                                                                \
      [left, right, dst](size_t beg, size_t end) -> void {                     \
        arithmetic_kernel<type>(left, right, dst, beg, end);                   \
      },                                                                       \
      0, element_size);                                                        \
  return;

void arithmetic(const float *left, float right, float *dst, size_t element_size,
                ArithmeticType type) {
  switch (type) {
  case ArithmeticType::kAdd:
    RUN_KERNEL(ArithmeticType::kAdd);
  case ArithmeticType::kMul:
    RUN_KERNEL(ArithmeticType::kMul);
  case ArithmeticType::kSub:
    RUN_KERNEL(ArithmeticType::kSub);
  case ArithmeticType::kDiv:
    TINY_LLM_CHECK(InvalidArgumentError, right != 0.F);
    right = 1.F / right;
    RUN_KERNEL(ArithmeticType::kMul);
  }
}
} // namespace tiny_llm::cpu
