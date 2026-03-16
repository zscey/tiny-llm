#pragma once

#include "tiny_llm/ops/cuda/cuda_op_common.hpp"

namespace tiny_llm::cuda {
/**
 * @brief RMSNorm.
 *
 * @param input [element_size, dim]
 * @param weight [dim]
 * @param dst [element_size, dim]
 * @param element_size
 * @param dim
 * @param eps
 */
void rms_norm(const float *input, const float *weight, float *dst,
              size_t element_size, uint32_t dim, float eps = 1e-6F);
} // namespace tiny_llm::cuda
