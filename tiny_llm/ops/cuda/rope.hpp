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
 * @brief Inplace operation.
 *
 * @param cos [max_len, dim / 2]
 * @param sin [max_len, dim / 2]
 * @param position_ids [batch, seq_len]
 * @param dst [batch, head_num, seq_end, dim], result write to [b, head_num,
 * seq_start: seq_start + seq_len, dim]
 * @param element_size
 * @param head_num
 * @param dim
 */
void apply_rope_inplace(const float *cos, const float *sin,
                        const uint32_t *position_ids, float *dst,
                        uint32_t batch, uint32_t head_num, uint32_t seq_len,
                        uint32_t dim, uint32_t seq_start, uint32_t seq_end);

/**
 * @brief [Paged] Rope.
 *
 * @param cos [max_len, dim / 2]
 * @param sin [max_len, dim / 2]
 * @param position_ids [total_queries], where `total_queries=sum(query_num[i])`
 * @param page_pool [num_page, num_head, page_size, head_dim]
 * @param block_table [num_requests, max_blocks]
 * @param seq_separator [num_requests]
 * @param cache_offsets [num_requests]
 * @param total_queries
 * @param num_requests
 * @param num_head
 * @param head_dim
 * @param max_blocks
 * @param page_size
 */
void apply_rope_inplace_paged(const float *cos, const float *sin,
                              const uint32_t *position_ids, float *page_pool,
                              const uint32_t *block_table,
                              const uint32_t *seq_separator,
                              const uint32_t *cache_offsets,
                              uint32_t total_queries, uint32_t num_requests,
                              uint32_t num_head, uint32_t head_dim,
                              uint32_t max_blocks, uint32_t page_size);
} // namespace tiny_llm::cuda
