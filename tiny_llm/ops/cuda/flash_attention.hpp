#pragma once

#include "tiny_llm/ops/cuda/cuda_op_common.hpp"

namespace tiny_llm::cuda {
enum class AttentionType : std::uint8_t {
  kNoMask,
  kCausalMask,
};

/**
 * @brief Flash attention.
 *
 * @param query [batch, q_head, q_length, dim]
 * @param key [batch, kv_head, kv_end, dim], use [batch, kv_head, :kv_length,
 * dim]
 * @param value [batch, kv_head, kv_end, dim], use [batch, kv_head, :kv_length,
 * dim]
 * @param dst [batch, q_head, q_length, dim]
 * @param batch
 * @param q_head
 * @param kv_head
 * @param q_length
 * @param kv_length
 * @param dim
 * @param kv_end
 * @param attn_type
 */
void flash_attn(const float *query, const float *key, const float *value,
                float *dst, uint32_t batch, uint32_t q_head, uint32_t kv_head,
                uint32_t q_length, uint32_t kv_length, uint32_t dim,
                uint32_t kv_end, AttentionType attn_type);

/**
 * @brief Paged flash attention.
 *
 * @param query [q_head, total_queries, head_dim], where
 * `total_queries=sum(q_lens[i])`
 * @param key_pool [num_page, kv_head, page_size, head_dim]
 * @param value_pool [num_page, kv_head, page_size, head_dim]
 * @param dst [q_head, total_queries, head_dim]
 * @param q_separator [num_requests + 1]
 * @param block_table [num_requests, max_blocks]
 * @param kv_lens [num_requests]
 * @param num_requests
 * @param q_head
 * @param kv_head
 * @param head_dim
 * @param max_blocks
 * @param page_size
 * @param max_q_len max(q_lens[i])
 * @param attn_type
 */
void flash_attn_paged(const float *query, const float *key_pool,
                      const float *value_pool, float *dst,
                      const uint32_t *q_separator, const uint32_t *block_table,
                      const uint32_t *kv_lens, uint32_t num_requests,
                      uint32_t q_head, uint32_t kv_head, uint32_t head_dim,
                      uint32_t max_blocks, uint32_t page_size,
                      uint32_t max_q_len, AttentionType attn_type);
} // namespace tiny_llm::cuda
