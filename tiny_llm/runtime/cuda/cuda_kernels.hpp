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
  uint32_t hidden_size{};

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
                 SliceLinearKernel>;
} // namespace tiny_llm::cuda
