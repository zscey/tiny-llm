#pragma once

#include "tiny_llm/ops/cuda/cuda_op_common.hpp"

namespace tiny_llm::cuda {
/**
 * @brief dst = matmul(input, weight) + bias
 *
 * @param input [m, d]
 * @param weight [n, d]
 * @param bias [n] or empty
 * @param dst [m, n]
 * @param m
 * @param d
 * @param n
 */
void gemm_row_major_plain(const float *input, const float *weight,
                          const float *bias, float *dst, uint32_t m, uint32_t d,
                          uint32_t n);

void gemm_row_major(const float *input, const float *weight, const float *bias,
                    float *dst, uint32_t m, uint32_t d, uint32_t n);

} // namespace tiny_llm::cuda
