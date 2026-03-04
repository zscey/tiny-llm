#pragma once

#include "tiny_llm/ops/cuda/cuda_op_common.hpp"

namespace tiny_llm::cuda {
/**
 * @brief Calculate the rope multiplier.
 *
 * @param cos_dst The length is `max_len * dim / 2`.
 * @param sin_dst The length is `max_len * dim / 2`.
 * @param max_len
 * @param dim
 * @param base
 */
void rope(float *cos_dst, float *sin_dst, uint32_t max_len, uint32_t dim,
          double base);

/**
 * @brief
 *
 * @param cos [max_len, dim / 2]
 * @param sin [max_len, dim / 2]
 * @param position_ids [element_size]
 * @param dst [element_size, head_num * dim]
 * @param element_size
 * @param head_num
 * @param dim
 */
void apply_rope_inplace(const float *cos, const float *sin,
                        const uint32_t *position_ids, float *dst,
                        size_t element_size, uint32_t head_num, uint32_t dim);
} // namespace tiny_llm::cuda
