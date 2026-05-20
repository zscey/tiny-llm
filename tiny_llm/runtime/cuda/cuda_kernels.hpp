#pragma once

#include "cuda_runtime.h"
#include "tiny_llm/graph/param.hpp"
#include "tiny_llm/runtime/common.hpp"
#include <cstddef>

namespace tiny_llm::cuda {
/**
 * 1. dtype_shape_infer(...): sets the `dtype` and `cur_shape` for each output
 * desc.
 * 2. execute(...): runs the corresponding kernel in the current thread context.
 */

class SiLUKernel {
public:
  SiLUParam param;

  size_t element_size{};

  /// @brief {input} -> {output}
  void dtype_shape_infer(const TensorDesc *const *input_descs,
                         TensorDesc *const *output_descs);
  void execute(const void *const *inputs, void *const *outputs) const;
};

class EmbeddingKernel {
public:
  EmbeddingParam param;

  size_t element_size{};

  /// @brief {input, weight} -> {output}
  void dtype_shape_infer(const TensorDesc *const *input_descs,
                         TensorDesc *const *output_descs);
  void execute(const void *const *inputs, void *const *outputs) const;
};

class RopeKernel {
public:
  RopeParam param;

  mutable bool initialized{};

  /// @brief {} -> {cos, sin}
  void dtype_shape_infer(const TensorDesc *const *input_descs,
                         TensorDesc *const *output_descs) const;
  void execute(const void *const *inputs, void *const *outputs);
};

class RMSNormKernel {
public:
  RMSNormParam param;

  size_t element_size{};

  /// @brief {input, weight} -> {output}
  void dtype_shape_infer(const TensorDesc *const *input_descs,
                         TensorDesc *const *output_descs);
  void execute(const void *const *inputs, void *const *outputs) const;
};

class AddKernel {
public:
  AddParam param;

  size_t element_size{};

  /// @brief {left, right} -> {output}
  void dtype_shape_infer(const TensorDesc *const *input_descs,
                         TensorDesc *const *output_descs);
  void execute(const void *const *inputs, void *const *outputs) const;
};

class MulKernel {
public:
  MulParam param;

  size_t element_size{};

  /// @brief {left, right} -> {output}
  void dtype_shape_infer(const TensorDesc *const *input_descs,
                         TensorDesc *const *output_descs);
  void execute(const void *const *inputs, void *const *outputs) const;
};

class LinearKernel {
public:
  LinearParam param;

  size_t element_size{};

  /// @brief {input, weight, [bias]} -{output}
  void dtype_shape_infer(const TensorDesc *const *input_descs,
                         TensorDesc *const *output_descs);
  void execute(const void *const *inputs, void *const *outputs) const;
};

// with kv-cache
class CausalAttentionKernel {
public:
  CausalAttentionParam param;

  uint32_t batch{};
  uint32_t seq_len{};

  bool is_prefill{true};
  mutable uint32_t cache_length{};
  float *q_cache{};
  float *k_cache{};
  float *v_cache{};
  float *o_cache{};

  /// @brief {hidden_state, cos, sin, pos_ids, q_weight, k_weight, v_weight,
  /// o_weight, [q_bias, k_bias, v_bias, o_bias]} -{output}
  void dtype_shape_infer(const TensorDesc *const *input_descs,
                         TensorDesc *const *output_descs);
  void execute(const void *const *inputs, void *const *outputs);
};

/// @brief The scheduling is outside of `CausalAttentionPagedKernel`, therefore
/// a class `OpMeta` is abstracted.
class CausalAttentionPagedKernel {
public:
  CausalAttentionParam param;

  struct OpMeta {
    bool is_prefill{};

    uint32_t page_size{};
    float *k_pool{};
    float *v_pool{};

    const uint32_t *block_table{};
    const uint32_t *seq_separator{};
    const uint32_t *cache_offsets{};
    const uint32_t *kv_lens{};
    uint32_t total_queries{};
    uint32_t num_requests{};
    uint32_t max_blocks{};
    uint32_t max_q_len{};
  };

  float *q_cache{};
  float *o_cache{};

  OpMeta meta;

  void set_meta(const OpMeta &m) { meta = m; }

  /// @brief {hidden_state, cos, sin, pos_ids, q_weight, k_weight, v_weight,
  /// o_weight, [q_bias, k_bias, v_bias, o_bias]} -{output}
  void dtype_shape_infer(const TensorDesc *const *input_descs,
                         TensorDesc *const *output_descs) const;
  void execute(const void *const *inputs, void *const *outputs) const;
};

class SliceLinearKernel {
public:
  SliceLinearParam param;

  uint32_t batch{};
  uint32_t q_end{};
  /// @brief {input, weight, [bias]} -{output}
  void dtype_shape_infer(const TensorDesc *const *input_descs,
                         TensorDesc *const *output_descs);
  void execute(const void *const *inputs, void *const *outputs) const;
};

using CudaKernel =
    std::variant<SiLUKernel, EmbeddingKernel, RopeKernel, RMSNormKernel,
                 AddKernel, MulKernel, LinearKernel, CausalAttentionKernel,
                 CausalAttentionPagedKernel, SliceLinearKernel>;
} // namespace tiny_llm::cuda
