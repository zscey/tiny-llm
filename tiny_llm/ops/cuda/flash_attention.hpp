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
 * @param query [batch, q_length, q_head * dim]
 * @param key [batch, kv_length, kv_head * dim]
 * @param value [batch, kv_length, kv_head * dim]
 * @param dst [batch, q_length, q_head * dim]
 * @param batch
 * @param q_length
 * @param kv_length
 * @param dim
 * @param q_head
 * @param kv_head
 * @param attn_type
 */
void flash_attn(const float *query, const float *key, const float *value,
                float *dst, uint32_t batch, uint32_t q_length,
                uint32_t kv_length, uint32_t dim, uint32_t q_head,
                uint32_t kv_head, AttentionType attn_type);
} // namespace tiny_llm::cuda
