#pragma once

#include "tiny_llm/ops/cuda/cuda_op_common.hpp"

namespace tiny_llm::cuda {
/**
 * @brief Calculate the rope multiplier.
 *
 * @param cos_dst The length is `max_len * dim`, and `cos_dst[s][d] =
 * cos_dst[s][d + dim / 2]`, where d is in the range [0, dim / 2).
 * @param sin_dst The length is `max_len * dim`, and `sin_dst[s][d] =
 * sin_dst[s][d + dim / 2]`, where d is in the range [0, dim / 2).
 * @param max_len
 * @param dim
 * @param base
 */
void rope(float *cos_dst, float *sin_dst, uint32_t max_len, uint32_t dim,
          double base);
} // namespace tiny_llm::cuda
