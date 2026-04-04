#pragma once

#include "tiny_llm/ops/arithmetic_type.hpp"
#include "tiny_llm/ops/cpu/cpu_op_common.hpp"

namespace tiny_llm::cpu {
/// @brief dst = left OP right, support inplace.
void arithmetic(const float *left, float right, float *dst, size_t element_size,
                ArithmeticType type);
} // namespace tiny_llm::cpu
