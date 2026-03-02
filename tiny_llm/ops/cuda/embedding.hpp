#pragma once

#include "tiny_llm/ops/cuda/cuda_op_common.hpp"

namespace tiny_llm::cuda {
/**
 * src: [n]
 * emb_weight: [m, dim]
 * dst [n, dim]
 */
void embedding(const uint32_t *src, const float *emb_weights, float *dst,
               uint32_t dim, size_t element_size);
} // namespace tiny_llm::cuda
