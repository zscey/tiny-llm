#pragma once

#include "tiny_llm/ops/cuda/cuda_op_common.hpp"

namespace tiny_llm::cuda {
/**
 * @brief Embedding
 *
 * @param src View as a shape [n].
 * @param emb_weights View as a shape [m, dim].
 * @param dst View as a shape [n, dim].
 * @param dim
 * @param element_size Equal to n.
 */
void embedding(const uint32_t *src, const float *emb_weights, float *dst,
               uint32_t dim, size_t element_size);
} // namespace tiny_llm::cuda
