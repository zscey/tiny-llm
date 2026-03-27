#pragma once

#include "cuda_runtime.h"
#include "tiny_llm/runtime/common.hpp"
#include <cstddef>
#include <variant>

namespace tiny_llm::cuda {
/**
 * 1. dtype_shape_infer(...): sets the `dtype` and `cur_shape` for each output
 * desc.
 * 2. execute(...): runs the corresponding kernel in the current thread context.
 */

class SiLUKernel {
public:
  bool inplace{};

  size_t element_size{};

  /// @brief {input} -> {output}
  void dtype_shape_infer(const TensorDesc *const *input_descs,
                         TensorDesc *const *output_descs);
  void execute(const void *const *inputs, void *const *outputs) const;
};

class EmbeddingKernel {
public:
  uint32_t num_embeddings{};
  uint32_t hidden_size{};

  size_t element_size{};

  /// @brief {input, weight} -> {output}
  void dtype_shape_infer(const TensorDesc *const *input_descs,
                         TensorDesc *const *output_descs);
  void execute(const void *const *inputs, void *const *outputs) const;
};

class RopeKernel {
public:
  uint32_t max_len{};
  uint32_t head_dim{};
  double theta{};
  bool pin{};

  mutable bool initialized{};

  /// @brief {} -> {cos, sin}
  void dtype_shape_infer(const TensorDesc *const *input_descs,
                         TensorDesc *const *output_descs) const;
  void execute(const void *const *inputs, void *const *outputs) const;
};

class RMSNormKernel {
public:
  uint32_t hidden_size{};
  float eps{};
  bool inplace{};

  size_t element_size{};

  /// @brief {input, weight} -> {output}
  void dtype_shape_infer(const TensorDesc *const *input_descs,
                         TensorDesc *const *output_descs);
  void execute(const void *const *inputs, void *const *outputs) const;
};

class AddKernel {
public:
  uint32_t left_idx{0};
  uint32_t right_idx{1};
  uint32_t out_idx{2};

  size_t element_size{};

  /// @brief {left, right} -> {output}
  void dtype_shape_infer(const TensorDesc *const *input_descs,
                         TensorDesc *const *output_descs);
  void execute(const void *const *inputs, void *const *outputs) const;
};

class MulKernel {
public:
  uint32_t left_idx{0};
  uint32_t right_idx{1};
  uint32_t out_idx{2};

  size_t element_size{};

  /// @brief {left, right} -> {output}
  void dtype_shape_infer(const TensorDesc *const *input_descs,
                         TensorDesc *const *output_descs);
  void execute(const void *const *inputs, void *const *outputs) const;
};

// class LinearKernel {
// public:
//   uint32_t in_dim{};
//   uint32_t out_dim{};
//   bool bias{};

//   size_t element_size{};

//   /// @brief {input, weight, [bias]} -{output}
//   void dtype_shape_infer(const TensorDesc *const *input_descs,
//                          TensorDesc *const *output_descs);
//   void execute(const void *const *inputs, void *const *outputs) const;
// };

// // with kv-cache
// class CausalAttentionKernel {
// public:
//   uint32_t head_dim{};
//   uint32_t q_head{};
//   uint32_t kv_head{};
//   bool bias{};

//   uint32_t batch{};
//   uint32_t seq_len{};
//   uint32_t hidden_size{};

//   bool is_prefill{};
//   mutable uint32_t cache_length{};
//   uint32_t max_length{};
//   float *q_cache{};
//   float *k_cache{};
//   float *v_cache{};
//   float *o_cache{};

//   /// @brief {hidden_state, cos, sin, pos_ids, q_weight, k_weight, v_weight,
//   /// [q_bias, k_bias, v_bias]} -{output}
//   void dtype_shape_infer(const TensorDesc *const *input_descs,
//                          TensorDesc *const *output_descs);
//   void execute(const void *const *inputs, void *const *outputs) const;
// };

using CudaKernel = std::variant<SiLUKernel, EmbeddingKernel, RopeKernel,
                                RMSNormKernel, AddKernel, MulKernel>;
} // namespace tiny_llm::cuda
