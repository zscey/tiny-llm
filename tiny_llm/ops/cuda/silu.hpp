#pragma once

#include "tiny_llm/ops/cuda/cuda_op_common.hpp"

namespace tiny_llm::cuda {
/// @brief Support inplace, which means `input == dst`.
void silu(const float *src, float *dst, size_t element_size);
} // namespace tiny_llm::cuda
