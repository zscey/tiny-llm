#pragma once

#include "tiny_llm/ops/cuda/cuda_op_common.hpp"

namespace tiny_llm::cuda {
/**
 * @brief dst = matmul(input, weight) + bias
 *
 * @param input [m, d], continuous
 * @param weight [n, d], continuous
 * @param bias [n] or empty, continuous
 * @param dst [m, n], continuous
 * @param m
 * @param d
 * @param n
 */
void gemm_row_major(const float *input, const float *weight, const float *bias,
                    float *dst, uint32_t m, uint32_t d, uint32_t n);

/**
 * @brief Linear + Transpose
 *
 * @param input [b, q, d], continuous
 * @param weight [out_head * out_d, d], continuous
 * @param bias [out_head * out_d] or empty, continuous
 * @param dst [b, out_head, q_end, out_d], result write to [b, out_head,
 * q_start: q_start + q, out_d]
 * @param b
 * @param q
 * @param d
 * @param out_head
 * @param out_d
 * @param q_start
 * @param q_end
 */
void gemm_row_major_lt(const float *input, const float *weight,
                       const float *bias, float *dst, uint32_t b, uint32_t q,
                       uint32_t d, uint32_t out_head, uint32_t out_d,
                       uint32_t q_start, uint32_t q_end);

/**
 * @brief Transpose + Linear
 *
 * @param input [b, h, q, d], continuous
 * @param weight [out_d, h * d], continuous
 * @param bias [out_d] or empty, continuous
 * @param dst [b, q, out_d], continuous
 * @param b
 * @param h
 * @param q
 * @param d
 * @param out_d
 */
void gemm_row_major_tl(const float *input, const float *weight,
                       const float *bias, float *dst, uint32_t b, uint32_t h,
                       uint32_t q, uint32_t d, uint32_t out_d);

/**
 * @brief Slice + Linear
 *
 * @param input [b, q_end, d], use its slice [b, q_start: q_start + q, d] as the
 * actual input
 * @param weight [n, d], continuous
 * @param bias [n] or empty, continuous
 * @param dst [b, q, n], continuous
 * @param b
 * @param q
 * @param d
 * @param n
 * @param q_start
 * @param q_end
 */
void gemm_row_major_sl(const float *input, const float *weight,
                       const float *bias, float *dst, uint32_t b, uint32_t q,
                       uint32_t d, uint32_t n, uint32_t q_start,
                       uint32_t q_end);

/**
 * @brief [Paged] Linear + Transpose
 *
 * @param inputs [total_queries, hidden_dim], where
 * `total_queries=sum(query_num[i])`
 * @param weight [num_head * head_dim, hidden_dim]
 * @param bias [num_head * head_dim]
 * @param page_pool [num_pages, num_head, page_size, head_dim]
 * @param block_table [num_requests, max_blocks]
 * @param seq_separator [num_requests]
 * @param cache_offsets [num_requests]
 * @param total_queries
 * @param num_requests
 * @param hidden_dim
 * @param num_head
 * @param head_dim
 * @param max_blocks
 * @param page_size
 */
void gemm_row_major_lt_paged(const float *inputs, const float *weight,
                             const float *bias, float *page_pool,
                             const uint32_t *block_table,
                             const uint32_t *seq_separator,
                             const uint32_t *cache_offsets,
                             uint32_t total_queries, uint32_t num_requests,
                             uint32_t hidden_dim, uint32_t num_head,
                             uint32_t head_dim, uint32_t max_blocks,
                             uint32_t page_size);

/**
 * @brief [Paged] Slice (-1) + Linear
 *
 * @param input [total_queries, d], where `total_queries=sum(query_num[i])`
 * @param weight [n, d]
 * @param bias [n]
 * @param dst [num_requests, n]
 * @param seq_separator [num_requests + 1]
 * @param num_requests
 * @param d
 * @param n
 */
void gemm_row_major_s1l_paged(const float *input, const float *weight,
                              const float *bias, float *dst,
                              const uint32_t *seq_separator,
                              uint32_t num_requests, uint32_t d, uint32_t n);
} // namespace tiny_llm::cuda
