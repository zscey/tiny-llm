#include "tiny_llm/runtime/cuda/cuda_kernels.hpp"
#include "tiny_llm/common/log_and_excepts.hpp"
#include "tiny_llm/ops/cuda/embedding.hpp"
#include "tiny_llm/ops/cuda/flash_attention.hpp"
#include "tiny_llm/ops/cuda/gemm.hpp"
#include "tiny_llm/ops/cuda/norm.hpp"
#include "tiny_llm/ops/cuda/rope.hpp"
#include "tiny_llm/ops/cuda/silu.hpp"
#include <numeric>

namespace tiny_llm::cuda {
namespace {
void check_desc_dtype_and_shape(const TensorDesc &desc, DataType dtype,
                                const std::vector<size_t> &cur_shape) {
  TINY_LLM_CHECK(desc.dtype == dtype);
  TINY_LLM_CHECK(desc.cur_shape == cur_shape);
}

void check_desc_dtype_and_shape(const TensorDesc &desc, DataType dtype,
                                size_t last_dim) {
  TINY_LLM_CHECK(desc.dtype == dtype);
  TINY_LLM_CHECK(!desc.cur_shape.empty());
  TINY_LLM_CHECK(desc.cur_shape.back() == last_dim);
}
} // namespace

void SiLUKernel::dtype_shape_infer(const TensorDesc *const *input_descs,
                                   TensorDesc *const *output_descs) {
  const auto *input_desc = input_descs[0];
  TINY_LLM_CHECK(input_desc->dtype == DataType::kFloat32);

  auto *output_desc = output_descs[0];
  output_desc->dtype = input_desc->dtype;
  output_desc->cur_shape = input_desc->cur_shape;
  element_size =
      input_desc->cur_shape.empty()
          ? 0
          : std::accumulate(input_desc->cur_shape.begin(),
                            input_desc->cur_shape.end(), 1,
                            [](auto a, auto b) -> auto { return a * b; });
}

void SiLUKernel::execute(const void *const *inputs,
                         void *const *outputs) const {
  silu(static_cast<const float *>(inputs[0]), static_cast<float *>(outputs[0]),
       element_size);
}

void EmbeddingKernel::dtype_shape_infer(const TensorDesc *const *input_descs,
                                        TensorDesc *const *output_descs) {
  const auto *input_desc = input_descs[0];
  TINY_LLM_CHECK(input_desc->dtype == DataType::kUint32);
  const auto *weight_desc = input_descs[1];
  check_desc_dtype_and_shape(*weight_desc, DataType::kFloat32,
                             {num_embeddings, hidden_size});

  auto *output_desc = output_descs[0];
  output_desc->dtype = weight_desc->dtype;
  output_desc->cur_shape = input_desc->cur_shape;
  output_desc->cur_shape.emplace_back(hidden_size);
  element_size =
      input_desc->cur_shape.empty()
          ? 0
          : std::accumulate(input_desc->cur_shape.begin(),
                            input_desc->cur_shape.end(), 1,
                            [](auto a, auto b) -> auto { return a * b; });
}

void EmbeddingKernel::execute(const void *const *inputs,
                              void *const *outputs) const {
  embedding(static_cast<const uint32_t *>(inputs[0]),
            static_cast<const float *>(inputs[1]),
            static_cast<float *>(outputs[0]), hidden_size, element_size);
}

// void RopeKernel::dtype_shape_infer(const TensorDesc *const * /*input_descs*/,
//                                    TensorDesc *const *output_descs) const {
//   for (uint32_t i = 0; i < 2; ++i) {
//     output_descs[i]->dtype = DataType::kFloat32;
//     output_descs[i]->cur_shape = {max_len, head_dim / 2};
//   }
// }

// void RopeKernel::execute(const void *const * /*inputs*/,
//                          void *const *outputs) const {
//   rope(static_cast<float *>(outputs[0]), static_cast<float *>(outputs[1]),
//        max_len, head_dim, theta);
// }

// void RMSNormKernel::dtype_shape_infer(const TensorDesc *const *input_descs,
//                                       TensorDesc *const *output_descs) {
//   const auto *input_desc = input_descs[0];
//   check_desc_dtype_and_shape(*input_desc, DataType::kFloat32, hidden_size);
//   const auto *weight_desc = input_descs[1];
//   check_desc_dtype_and_shape(*weight_desc, DataType::kFloat32,
//                              std::vector<size_t>{hidden_size});

//   auto *output_desc = output_descs[0];
//   output_desc->dtype = input_desc->dtype;
//   output_desc->cur_shape = input_desc->cur_shape;
//   element_size = std::max(
//       1, std::accumulate(input_desc->cur_shape.begin(),
//                          std::prev(input_desc->cur_shape.end()), 1,
//                          [](auto a, auto b) -> auto { return a * b; }));
// }

// void RMSNormKernel::execute(const void *const *inputs,
//                             void *const *outputs) const {
//   rms_norm(static_cast<const float *>(inputs[0]),
//            static_cast<const float *>(inputs[1]),
//            static_cast<float *>(outputs[0]), element_size, hidden_size, eps);
// }

// void LinearKernel::dtype_shape_infer(const TensorDesc *const *input_descs,
//                                      TensorDesc *const *output_descs) {
//   const auto *input_desc = input_descs[0];
//   check_desc_dtype_and_shape(*input_desc, DataType::kFloat32, in_dim);
//   const auto *weight_desc = input_descs[1];
//   check_desc_dtype_and_shape(*weight_desc, DataType::kFloat32,
//                              {out_dim, in_dim});
//   if (bias) {
//     const auto *bias_desc = input_descs[2];
//     check_desc_dtype_and_shape(*bias_desc, DataType::kFloat32,
//                                std::vector<size_t>{out_dim});
//   }

//   auto *output_desc = output_descs[0];
//   output_desc->dtype = input_desc->dtype;
//   output_desc->cur_shape = input_desc->cur_shape;
//   output_desc->cur_shape.back() = out_dim;
//   element_size = std::max(
//       1, std::accumulate(input_desc->cur_shape.begin(),
//                          std::prev(input_desc->cur_shape.end()), 1,
//                          [](auto a, auto b) -> auto { return a * b; }));
// }

// void LinearKernel::execute(const void *const *inputs,
//                            void *const *outputs) const {
//   const auto *bias_ptr = bias ? static_cast<const float *>(inputs[2]) :
//   nullptr; gemm_row_major(static_cast<const float *>(inputs[0]),
//                  static_cast<const float *>(inputs[1]), bias_ptr,
//                  static_cast<float *>(outputs[0]), element_size, in_dim,
//                  out_dim);
// }

// void CausalAttentionKernel::dtype_shape_infer(
//     const TensorDesc *const *input_descs, TensorDesc *const *output_descs) {
//   auto q_dim = static_cast<size_t>(q_head) * head_dim;
//   auto kv_dim = static_cast<size_t>(kv_head) * head_dim;

//   const auto *h_desc = input_descs[0];
//   TINY_LLM_CHECK(h_desc->cur_shape.size() == 3);
//   check_desc_dtype_and_shape(*h_desc, DataType::kFloat32, q_dim);
//   const auto *cos_desc = input_descs[1];
//   TINY_LLM_CHECK(cos_desc->cur_shape.size() == 2);
//   check_desc_dtype_and_shape(*cos_desc, DataType::kFloat32, head_dim / 2);
//   const auto *sin_desc = input_descs[2];
//   check_desc_dtype_and_shape(*sin_desc, DataType::kFloat32,
//                              cos_desc->cur_shape);
//   const auto *pos_desc = input_descs[3];
//   check_desc_dtype_and_shape(
//       *pos_desc, DataType::kUint32,
//       {h_desc->cur_shape.begin(), std::prev(h_desc->cur_shape.end())});
//   // q, k, v, o weight
//   check_desc_dtype_and_shape(*input_descs[4], DataType::kFloat32,
//                              {q_dim, q_dim});
//   check_desc_dtype_and_shape(*input_descs[5], DataType::kFloat32,
//                              {kv_dim, q_dim});
//   check_desc_dtype_and_shape(*input_descs[6], DataType::kFloat32,
//                              {kv_dim, q_dim});
//   check_desc_dtype_and_shape(*input_descs[7], DataType::kFloat32,
//                              {q_dim, q_dim});
//   if (bias) {
//     // q, k, v, o bias
//     check_desc_dtype_and_shape(*input_descs[8], DataType::kFloat32,
//                                std::vector<size_t>{q_dim});
//     check_desc_dtype_and_shape(*input_descs[9], DataType::kFloat32,
//                                std::vector<size_t>{kv_dim});
//     check_desc_dtype_and_shape(*input_descs[10], DataType::kFloat32,
//                                std::vector<size_t>{kv_dim});
//     check_desc_dtype_and_shape(*input_descs[11], DataType::kFloat32,
//                                std::vector<size_t>{q_dim});
//   }

//   auto *output_desc = output_descs[0];
//   output_desc->dtype = h_desc->dtype;
//   output_desc->cur_shape = h_desc->cur_shape;

//   batch = h_desc->cur_shape.at(0);
//   // attention kernel with mask is not implemented, thus the batch must be 1.
//   TINY_LLM_CHECK(batch == 1);
//   seq_len = h_desc->cur_shape.at(1);
//   hidden_size = h_desc->cur_shape.at(2);
// }

// void CausalAttentionKernel::execute(const void *const *inputs,
//                                     void *const *outputs) const {
//   const auto *hidden_state = static_cast<const float *>(inputs[0]);
//   const auto *cos = static_cast<const float *>(inputs[1]);
//   const auto *sin = static_cast<const float *>(inputs[2]);
//   const auto *pos_ids = static_cast<const uint32_t *>(inputs[3]);
//   const auto *q_weight = static_cast<const float *>(inputs[4]);
//   const auto *k_weight = static_cast<const float *>(inputs[5]);
//   const auto *v_weight = static_cast<const float *>(inputs[6]);
//   const auto *o_weight = static_cast<const float *>(inputs[7]);
//   const auto *q_bias = bias ? static_cast<const float *>(inputs[8]) :
//   nullptr; const auto *k_bias = bias ? static_cast<const float *>(inputs[9])
//   : nullptr; const auto *v_bias = bias ? static_cast<const float
//   *>(inputs[10]) : nullptr; const auto *o_bias = bias ? static_cast<const
//   float *>(inputs[11]) : nullptr;

//   cuda::gemm_row_major_lt(hidden_state, q_weight, q_bias, q_cache, batch,
//                           seq_len, hidden_size, q_head, head_dim, 0,
//                           seq_len);
//   cuda::apply_rope_inplace(cos, sin, pos_ids, q_cache, batch, q_head,
//   seq_len,
//                            head_dim, 0, seq_len);
//   cuda::gemm_row_major_lt(hidden_state, k_weight, k_bias, k_cache, batch,
//                           seq_len, hidden_size, kv_head, head_dim,
//                           cache_length, max_length);
//   cuda::apply_rope_inplace(cos, sin, pos_ids, k_cache, batch, kv_head,
//   seq_len,
//                            head_dim, cache_length, max_length);
//   cuda::gemm_row_major_lt(hidden_state, v_weight, v_bias, v_cache, batch,
//                           seq_len, hidden_size, kv_head, head_dim,
//                           cache_length, max_length);

//   cache_length = is_prefill ? seq_len : cache_length + 1;
//   cuda::flash_attn(q_cache, k_cache, v_cache, o_cache, batch, q_head,
//   kv_head,
//                    seq_len, cache_length, head_dim, max_length,
//                    AttentionType::kCausalMask);
//   cuda::gemm_row_major_tl(o_cache, o_weight, o_bias,
//                           static_cast<float *>(outputs[0]), batch, q_head,
//                           seq_len, head_dim, hidden_size);
// }
} // namespace tiny_llm::cuda
