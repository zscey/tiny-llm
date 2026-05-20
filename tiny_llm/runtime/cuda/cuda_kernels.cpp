#include "tiny_llm/runtime/cuda/cuda_kernels.hpp"
#include "tiny_llm/common/checks.hpp"
#include "tiny_llm/common/exception.hpp"
#include "tiny_llm/ops/cuda/arithmetic.hpp"
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
  TINY_LLM_CHECK(InvalidArgumentError, desc.dtype == dtype);
  TINY_LLM_CHECK(InvalidArgumentError, desc.cur_shape == cur_shape);
}

void check_desc_dtype_and_shape(const TensorDesc &desc, DataType dtype,
                                size_t last_dim) {
  TINY_LLM_CHECK(InvalidArgumentError, desc.dtype == dtype);
  TINY_LLM_CHECK(InvalidArgumentError, !desc.cur_shape.empty());
  TINY_LLM_CHECK(InvalidArgumentError, desc.cur_shape.back() == last_dim);
}

auto element_num(const std::vector<size_t> &shape) -> size_t {
  return shape.empty()
             ? 0
             : std::accumulate(
                   shape.begin(), shape.end(), 1,
                   [](const auto &a, const auto &b) -> auto { return a * b; });
}
} // namespace

void SiLUKernel::dtype_shape_infer(const TensorDesc *const *input_descs,
                                   TensorDesc *const *output_descs) {
  const auto *input_desc = input_descs[0];
  TINY_LLM_CHECK(RuntimeError, input_desc->dtype == DataType::kFloat32);

  auto *output_desc = output_descs[0];
  output_desc->dtype = input_desc->dtype;
  output_desc->cur_shape = input_desc->cur_shape;
  element_size = element_num(input_desc->cur_shape);
}

void SiLUKernel::execute(const void *const *inputs,
                         void *const *outputs) const {
  silu(static_cast<const float *>(inputs[0]), static_cast<float *>(outputs[0]),
       element_size);
}

void EmbeddingKernel::dtype_shape_infer(const TensorDesc *const *input_descs,
                                        TensorDesc *const *output_descs) {
  const auto *input_desc = input_descs[0];
  TINY_LLM_CHECK(RuntimeError, input_desc->dtype == DataType::kUint32);
  const auto *weight_desc = input_descs[1];
  check_desc_dtype_and_shape(*weight_desc, DataType::kFloat32,
                             {param.num_embeddings, param.hidden_size});

  auto *output_desc = output_descs[0];
  output_desc->dtype = weight_desc->dtype;
  output_desc->cur_shape = input_desc->cur_shape;
  output_desc->cur_shape.emplace_back(param.hidden_size);
  element_size = element_num(input_desc->cur_shape);
}

void EmbeddingKernel::execute(const void *const *inputs,
                              void *const *outputs) const {
  embedding(static_cast<const uint32_t *>(inputs[0]),
            static_cast<const float *>(inputs[1]),
            static_cast<float *>(outputs[0]), param.hidden_size, element_size);
}

void RopeKernel::dtype_shape_infer(const TensorDesc *const * /*input_descs*/,
                                   TensorDesc *const *output_descs) const {
  for (uint32_t i = 0; i < 2; ++i) {
    output_descs[i]->dtype = DataType::kFloat32;
    output_descs[i]->cur_shape = {param.max_len, param.head_dim / 2};
  }
}

void RopeKernel::execute(const void *const * /*inputs*/, void *const *outputs) {
  if (param.pin && initialized) {
    return;
  }
  rope(static_cast<float *>(outputs[0]), static_cast<float *>(outputs[1]),
       param.max_len, param.head_dim, param.theta);
  initialized = true;
}

void RMSNormKernel::dtype_shape_infer(const TensorDesc *const *input_descs,
                                      TensorDesc *const *output_descs) {
  const auto *input_desc = input_descs[0];
  check_desc_dtype_and_shape(*input_desc, DataType::kFloat32,
                             param.hidden_size);
  const auto *weight_desc = input_descs[1];
  check_desc_dtype_and_shape(*weight_desc, DataType::kFloat32,
                             std::vector<size_t>{param.hidden_size});

  auto *output_desc = output_descs[0];
  output_desc->dtype = input_desc->dtype;
  output_desc->cur_shape = input_desc->cur_shape;
  element_size = std::accumulate(input_desc->cur_shape.begin(),
                                 std::prev(input_desc->cur_shape.end()), 1,
                                 [](auto a, auto b) -> auto { return a * b; });
}

void RMSNormKernel::execute(const void *const *inputs,
                            void *const *outputs) const {
  rms_norm(static_cast<const float *>(inputs[0]),
           static_cast<const float *>(inputs[1]),
           static_cast<float *>(outputs[0]), element_size, param.hidden_size,
           param.eps);
}

void AddKernel::dtype_shape_infer(const TensorDesc *const *input_descs,
                                  TensorDesc *const *output_descs) {
  const auto *left_desc = input_descs[0];
  const auto *right_desc = input_descs[1];
  TINY_LLM_CHECK(RuntimeError, left_desc->dtype == right_desc->dtype);
  TINY_LLM_CHECK(RuntimeError, left_desc->cur_shape == right_desc->cur_shape);

  auto *output_desc = output_descs[0];
  output_desc->dtype = left_desc->dtype;
  output_desc->cur_shape = left_desc->cur_shape;
  element_size = element_num(left_desc->cur_shape);
}

void AddKernel::execute(const void *const *inputs, void *const *outputs) const {
  arithmetic(static_cast<const float *>(inputs[0]),
             static_cast<const float *>(inputs[1]),
             static_cast<float *>(outputs[0]), element_size,
             ArithmeticType::kAdd);
}

void MulKernel::dtype_shape_infer(const TensorDesc *const *input_descs,
                                  TensorDesc *const *output_descs) {
  const auto *left_desc = input_descs[0];
  const auto *right_desc = input_descs[1];
  TINY_LLM_CHECK(RuntimeError, left_desc->dtype == right_desc->dtype);
  TINY_LLM_CHECK(RuntimeError, left_desc->cur_shape == right_desc->cur_shape);

  auto *output_desc = output_descs[0];
  output_desc->dtype = left_desc->dtype;
  output_desc->cur_shape = left_desc->cur_shape;
  element_size = element_num(left_desc->cur_shape);
}

void MulKernel::execute(const void *const *inputs, void *const *outputs) const {
  arithmetic(static_cast<const float *>(inputs[0]),
             static_cast<const float *>(inputs[1]),
             static_cast<float *>(outputs[0]), element_size,
             ArithmeticType::kMul);
}

void LinearKernel::dtype_shape_infer(const TensorDesc *const *input_descs,
                                     TensorDesc *const *output_descs) {
  const auto *input_desc = input_descs[0];
  check_desc_dtype_and_shape(*input_desc, DataType::kFloat32, param.in_dim);
  const auto *weight_desc = input_descs[1];
  check_desc_dtype_and_shape(*weight_desc, DataType::kFloat32,
                             {param.out_dim, param.in_dim});
  if (param.bias) {
    const auto *bias_desc = input_descs[2];
    check_desc_dtype_and_shape(*bias_desc, DataType::kFloat32,
                               std::vector<size_t>{param.out_dim});
  }

  auto *output_desc = output_descs[0];
  output_desc->dtype = input_desc->dtype;
  output_desc->cur_shape = input_desc->cur_shape;
  output_desc->cur_shape.back() = param.out_dim;
  element_size = std::accumulate(input_desc->cur_shape.begin(),
                                 std::prev(input_desc->cur_shape.end()), 1,
                                 [](auto a, auto b) -> auto { return a * b; });
}

void LinearKernel::execute(const void *const *inputs,
                           void *const *outputs) const {
  const auto *bias_ptr =
      param.bias ? static_cast<const float *>(inputs[2]) : nullptr;
  gemm_row_major(static_cast<const float *>(inputs[0]),
                 static_cast<const float *>(inputs[1]), bias_ptr,
                 static_cast<float *>(outputs[0]), element_size, param.in_dim,
                 param.out_dim);
}

void CausalAttentionKernel::dtype_shape_infer(
    const TensorDesc *const *input_descs, TensorDesc *const *output_descs) {
  auto q_dim = static_cast<size_t>(param.q_head) * param.head_dim;
  auto kv_dim = static_cast<size_t>(param.kv_head) * param.head_dim;

  const auto *h_desc = input_descs[0];
  TINY_LLM_CHECK(RuntimeError, h_desc->cur_shape.size() == 3);
  check_desc_dtype_and_shape(*h_desc, DataType::kFloat32, q_dim);
  const auto *cos_desc = input_descs[1];
  TINY_LLM_CHECK(RuntimeError, cos_desc->cur_shape.size() == 2);
  check_desc_dtype_and_shape(*cos_desc, DataType::kFloat32, param.head_dim / 2);
  const auto *sin_desc = input_descs[2];
  check_desc_dtype_and_shape(*sin_desc, DataType::kFloat32,
                             cos_desc->cur_shape);
  const auto *pos_desc = input_descs[3];
  check_desc_dtype_and_shape(
      *pos_desc, DataType::kUint32,
      {h_desc->cur_shape.begin(), std::prev(h_desc->cur_shape.end())});
  // q, k, v, o weight
  check_desc_dtype_and_shape(*input_descs[4], DataType::kFloat32,
                             {q_dim, q_dim});
  check_desc_dtype_and_shape(*input_descs[5], DataType::kFloat32,
                             {kv_dim, q_dim});
  check_desc_dtype_and_shape(*input_descs[6], DataType::kFloat32,
                             {kv_dim, q_dim});
  check_desc_dtype_and_shape(*input_descs[7], DataType::kFloat32,
                             {q_dim, q_dim});
  if (param.bias) {
    // q, k, v, o bias
    check_desc_dtype_and_shape(*input_descs[8], DataType::kFloat32,
                               std::vector<size_t>{q_dim});
    check_desc_dtype_and_shape(*input_descs[9], DataType::kFloat32,
                               std::vector<size_t>{kv_dim});
    check_desc_dtype_and_shape(*input_descs[10], DataType::kFloat32,
                               std::vector<size_t>{kv_dim});
    check_desc_dtype_and_shape(*input_descs[11], DataType::kFloat32,
                               std::vector<size_t>{q_dim});
  }

  auto *output_desc = output_descs[0];
  output_desc->dtype = h_desc->dtype;
  output_desc->cur_shape = h_desc->cur_shape;

  batch = h_desc->cur_shape.at(0);
  // attention kernel with mask is not implemented, thus the batch must be 1.
  TINY_LLM_CHECK(RuntimeError, batch == 1);
  seq_len = h_desc->cur_shape.at(1);
  if (!is_prefill) {
    TINY_LLM_CHECK(RuntimeError, seq_len == 1);
  }
}

void CausalAttentionKernel::execute(const void *const *inputs,
                                    void *const *outputs) {
  cache_length = is_prefill ? 0 : cache_length;
  auto new_cache_length = seq_len + cache_length;
  TINY_LLM_CHECK(RuntimeError, new_cache_length <= param.max_len);

  const auto *hidden_state = static_cast<const float *>(inputs[0]);
  const auto *cos = static_cast<const float *>(inputs[1]);
  const auto *sin = static_cast<const float *>(inputs[2]);
  const auto *pos_ids = static_cast<const uint32_t *>(inputs[3]);
  const auto *q_weight = static_cast<const float *>(inputs[4]);
  const auto *k_weight = static_cast<const float *>(inputs[5]);
  const auto *v_weight = static_cast<const float *>(inputs[6]);
  const auto *o_weight = static_cast<const float *>(inputs[7]);
  const auto *q_bias =
      param.bias ? static_cast<const float *>(inputs[8]) : nullptr;
  const auto *k_bias =
      param.bias ? static_cast<const float *>(inputs[9]) : nullptr;
  const auto *v_bias =
      param.bias ? static_cast<const float *>(inputs[10]) : nullptr;
  const auto *o_bias =
      param.bias ? static_cast<const float *>(inputs[11]) : nullptr;

  cuda::gemm_row_major_lt(hidden_state, q_weight, q_bias, q_cache, batch,
                          seq_len, param.q_head * param.head_dim, param.q_head,
                          param.head_dim, 0, seq_len);
  cuda::apply_rope_inplace(cos, sin, pos_ids, q_cache, batch, param.q_head,
                           seq_len, param.head_dim, 0, seq_len);
  cuda::gemm_row_major_lt(hidden_state, k_weight, k_bias, k_cache, batch,
                          seq_len, param.q_head * param.head_dim, param.kv_head,
                          param.head_dim, cache_length, param.max_len);
  cuda::apply_rope_inplace(cos, sin, pos_ids, k_cache, batch, param.kv_head,
                           seq_len, param.head_dim, cache_length,
                           param.max_len);
  cuda::gemm_row_major_lt(hidden_state, v_weight, v_bias, v_cache, batch,
                          seq_len, param.q_head * param.head_dim, param.kv_head,
                          param.head_dim, cache_length, param.max_len);
  auto attn_type =
      is_prefill ? AttentionType::kCausalMask : AttentionType::kNoMask;
  cuda::flash_attn(q_cache, k_cache, v_cache, o_cache, batch, param.q_head,
                   param.kv_head, seq_len, new_cache_length, param.head_dim,
                   param.max_len, attn_type);
  cuda::gemm_row_major_tl(
      o_cache, o_weight, o_bias, static_cast<float *>(outputs[0]), batch,
      param.q_head, seq_len, param.head_dim, param.q_head * param.head_dim);
  cache_length = new_cache_length;
}

void CausalAttentionPagedKernel::dtype_shape_infer(
    const TensorDesc *const *input_descs,
    TensorDesc *const *output_descs) const {
  auto q_dim = static_cast<size_t>(param.q_head) * param.head_dim;
  auto kv_dim = static_cast<size_t>(param.kv_head) * param.head_dim;

  const auto *h_desc = input_descs[0];
  TINY_LLM_CHECK(RuntimeError, h_desc->cur_shape.size() == 3);
  TINY_LLM_CHECK(RuntimeError, h_desc->cur_shape.at(0) == 1);
  check_desc_dtype_and_shape(*h_desc, DataType::kFloat32, q_dim);
  const auto *cos_desc = input_descs[1];
  TINY_LLM_CHECK(RuntimeError, cos_desc->cur_shape.size() == 2);
  check_desc_dtype_and_shape(*cos_desc, DataType::kFloat32, param.head_dim / 2);
  const auto *sin_desc = input_descs[2];
  check_desc_dtype_and_shape(*sin_desc, DataType::kFloat32,
                             cos_desc->cur_shape);
  const auto *pos_desc = input_descs[3];
  check_desc_dtype_and_shape(
      *pos_desc, DataType::kUint32,
      {h_desc->cur_shape.begin(), std::prev(h_desc->cur_shape.end())});
  // q, k, v, o weight
  check_desc_dtype_and_shape(*input_descs[4], DataType::kFloat32,
                             {q_dim, q_dim});
  check_desc_dtype_and_shape(*input_descs[5], DataType::kFloat32,
                             {kv_dim, q_dim});
  check_desc_dtype_and_shape(*input_descs[6], DataType::kFloat32,
                             {kv_dim, q_dim});
  check_desc_dtype_and_shape(*input_descs[7], DataType::kFloat32,
                             {q_dim, q_dim});
  if (param.bias) {
    // q, k, v, o bias
    check_desc_dtype_and_shape(*input_descs[8], DataType::kFloat32,
                               std::vector<size_t>{q_dim});
    check_desc_dtype_and_shape(*input_descs[9], DataType::kFloat32,
                               std::vector<size_t>{kv_dim});
    check_desc_dtype_and_shape(*input_descs[10], DataType::kFloat32,
                               std::vector<size_t>{kv_dim});
    check_desc_dtype_and_shape(*input_descs[11], DataType::kFloat32,
                               std::vector<size_t>{q_dim});
  }

  auto *output_desc = output_descs[0];
  output_desc->dtype = h_desc->dtype;
  output_desc->cur_shape = h_desc->cur_shape;
}

void CausalAttentionPagedKernel::execute(const void *const *inputs,
                                         void *const *outputs) const {
  const auto *hidden_state = static_cast<const float *>(inputs[0]);
  const auto *cos = static_cast<const float *>(inputs[1]);
  const auto *sin = static_cast<const float *>(inputs[2]);
  const auto *pos_ids = static_cast<const uint32_t *>(inputs[3]);
  const auto *q_weight = static_cast<const float *>(inputs[4]);
  const auto *k_weight = static_cast<const float *>(inputs[5]);
  const auto *v_weight = static_cast<const float *>(inputs[6]);
  const auto *o_weight = static_cast<const float *>(inputs[7]);
  const auto *q_bias =
      param.bias ? static_cast<const float *>(inputs[8]) : nullptr;
  const auto *k_bias =
      param.bias ? static_cast<const float *>(inputs[9]) : nullptr;
  const auto *v_bias =
      param.bias ? static_cast<const float *>(inputs[10]) : nullptr;
  const auto *o_bias =
      param.bias ? static_cast<const float *>(inputs[11]) : nullptr;

  cuda::gemm_row_major_lt(hidden_state, q_weight, q_bias, q_cache, 1,
                          meta.total_queries, param.q_head * param.head_dim,
                          param.q_head, param.head_dim, 0, meta.total_queries);
  cuda::apply_rope_inplace(cos, sin, pos_ids, q_cache, 1, param.q_head,
                           meta.total_queries, param.head_dim, 0,
                           meta.total_queries);
  cuda::gemm_row_major_lt_paged(
      hidden_state, k_weight, k_bias, meta.k_pool, meta.block_table,
      meta.seq_separator, meta.cache_offsets, meta.total_queries,
      meta.num_requests, param.q_head * param.head_dim, param.kv_head,
      param.head_dim, meta.max_blocks, meta.page_size);
  cuda::apply_rope_inplace_paged(
      cos, sin, pos_ids, meta.k_pool, meta.block_table, meta.seq_separator,
      meta.cache_offsets, meta.total_queries, meta.num_requests, param.kv_head,
      param.head_dim, meta.max_blocks, meta.page_size);
  cuda::gemm_row_major_lt_paged(
      hidden_state, v_weight, v_bias, meta.v_pool, meta.block_table,
      meta.seq_separator, meta.cache_offsets, meta.total_queries,
      meta.num_requests, param.q_head * param.head_dim, param.kv_head,
      param.head_dim, meta.max_blocks, meta.page_size);
  cuda::flash_attn_paged(
      q_cache, meta.k_pool, meta.v_pool, o_cache, meta.seq_separator,
      meta.block_table, meta.kv_lens, meta.num_requests, param.q_head,
      param.kv_head, param.head_dim, meta.max_blocks, meta.page_size,
      meta.max_q_len,
      meta.is_prefill ? AttentionType::kCausalMask : AttentionType::kNoMask);
  cuda::gemm_row_major_tl(o_cache, o_weight, o_bias,
                          static_cast<float *>(outputs[0]), 1, param.q_head,
                          meta.total_queries, param.head_dim,
                          param.q_head * param.head_dim);
}

void SliceLinearKernel::dtype_shape_infer(const TensorDesc *const *input_descs,
                                          TensorDesc *const *output_descs) {
  const auto *input_desc = input_descs[0];
  TINY_LLM_CHECK(RuntimeError, input_desc->cur_shape.size() == 3);
  check_desc_dtype_and_shape(*input_desc, DataType::kFloat32, param.in_dim);
  const auto *weight_desc = input_descs[1];
  check_desc_dtype_and_shape(*weight_desc, DataType::kFloat32,
                             {param.out_dim, param.in_dim});
  if (param.bias) {
    const auto *bias_desc = input_descs[2];
    check_desc_dtype_and_shape(*bias_desc, DataType::kFloat32,
                               std::vector<size_t>{param.out_dim});
  }

  auto *output_desc = output_descs[0];
  output_desc->dtype = input_desc->dtype;
  batch = static_cast<uint32_t>(input_desc->cur_shape[0]);
  q_end = static_cast<uint32_t>(input_desc->cur_shape[1]);
  TINY_LLM_CHECK(RuntimeError, q_end >= param.only_last_q);
  output_desc->cur_shape = {batch, param.only_last_q, param.out_dim};
}

void SliceLinearKernel::execute(const void *const *inputs,
                                void *const *outputs) const {
  const auto *bias_ptr =
      param.bias ? static_cast<const float *>(inputs[2]) : nullptr;
  gemm_row_major_sl(static_cast<const float *>(inputs[0]),
                    static_cast<const float *>(inputs[1]), bias_ptr,
                    static_cast<float *>(outputs[0]), batch, param.only_last_q,
                    param.in_dim, param.out_dim, q_end - param.only_last_q,
                    q_end);
}
} // namespace tiny_llm::cuda
