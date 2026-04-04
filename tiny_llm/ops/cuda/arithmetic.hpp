#pragma once

#include "tiny_llm/ops/arithmetic_type.hpp"
#include "tiny_llm/ops/cuda/cuda_op_common.hpp"

namespace tiny_llm::cuda {

/// @brief dst = left OP right, support inplace.
void arithmetic(const float *left, const float *right, float *dst,
                size_t element_size, ArithmeticType type);
} // namespace tiny_llm::cuda
