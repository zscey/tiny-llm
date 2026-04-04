#include "tiny_llm/ops/cpu/arithmetic.hpp"

namespace tiny_llm::cpu {
namespace {
template <ArithmeticType T>
void arithmetic_kernel(const float *left, float right, float *dst, size_t beg,
                       size_t end) {
  if constexpr (T == ArithmeticType::kAdd) {
    while (beg < end) {
      dst[beg] = left[beg] + right;
      ++beg;
    }
  } else if constexpr (T == ArithmeticType::kMul) {
    while (beg < end) {
      dst[beg] = left[beg] * right;
      ++beg;
    }
  } else if constexpr (T == ArithmeticType::kSub) {
    while (beg < end) {
      dst[beg] = left[beg] - right;
      ++beg;
    }
  } else if constexpr (T == ArithmeticType::kDiv) {
    while (beg < end) {
      dst[beg] = left[beg] / right;
      ++beg;
    }
  }
}
} // namespace

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define CASE_OF_TYPE(type)                                                     \
  case type:                                                                   \
    parallel_run(                                                              \
        [left, right, dst](size_t beg, size_t end) -> void {                   \
          arithmetic_kernel<type>(left, right, dst, beg, end);                 \
        },                                                                     \
        0, element_size);                                                      \
    return;

void arithmetic(const float *left, float right, float *dst, size_t element_size,
                ArithmeticType type) {
  switch (type) {
    CASE_OF_TYPE(ArithmeticType::kAdd);
    CASE_OF_TYPE(ArithmeticType::kMul);
    CASE_OF_TYPE(ArithmeticType::kSub);
    CASE_OF_TYPE(ArithmeticType::kDiv);
  }
}
} // namespace tiny_llm::cpu
