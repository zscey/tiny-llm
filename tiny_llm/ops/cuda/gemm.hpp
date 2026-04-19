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
void gemm(const float *input, const float *weight, const float *bias,
          float *dst, uint32_t m, uint32_t d, uint32_t n);

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
 * @brief
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
 * @brief
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
void gemm_row_major_l(const float *input, const float *weight,
                      const float *bias, float *dst, uint32_t b, uint32_t q,
                      uint32_t d, uint32_t n, uint32_t q_start, uint32_t q_end);
} // namespace tiny_llm::cuda
