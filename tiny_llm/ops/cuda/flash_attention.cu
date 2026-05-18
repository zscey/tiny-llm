#include "cccl/cuda/warp"
#include "cub/cub.cuh"
#include "math_constants.h"
#include "tiny_llm/common/checks.hpp"
#include "tiny_llm/common/exception.hpp"
#include "tiny_llm/device_managers/cuda/cuda_device_infos.hpp"
#include "tiny_llm/ops/cuda/flash_attention.hpp"

namespace tiny_llm::cuda {
// NOLINTBEGIN(cppcoreguidelines-pro-bounds-constant-array-index)
namespace {
constexpr uint32_t kWarpSize = 32;
constexpr uint32_t kWarpNumPerBlock = 16;
constexpr uint32_t kThreadNumPerBlock = kWarpNumPerBlock * kWarpSize;
constexpr uint32_t kWarpIterPerTileQ = 4;
constexpr uint32_t kTileQ = kWarpIterPerTileQ * kWarpNumPerBlock;
constexpr uint32_t kThreadIterPerKV = 2;
constexpr uint32_t kTileKV = kThreadIterPerKV * kWarpSize;

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define GET_PADDED_DIM(dim)                                                    \
  ((((dim) + kWarpSize - 1) / kWarpSize * kWarpSize) + 1)

__device__ auto get_shift(uint32_t batch_idx, uint32_t head_idx,
                          uint32_t seq_idx, uint32_t head_num, uint32_t seq_len,
                          uint32_t dim) -> size_t {
  return ((((static_cast<size_t>(batch_idx) * head_num) + head_idx) * seq_len) +
          seq_idx) *
         dim;
}

__device__ void fetch_data(const float *from, float *to, uint32_t row_num,
                           uint32_t row_length, uint32_t from_step,
                           uint32_t to_step, float scale = 1.F) {
  for (uint32_t row = threadIdx.x / kWarpSize; row < row_num;
       row += kWarpNumPerBlock) {
    const auto *cur_from = from + (static_cast<size_t>(row) * from_step);
    auto *cur_to = to + (static_cast<size_t>(row) * to_step);
    for (uint32_t col = threadIdx.x % kWarpSize; col < row_length;
         col += kWarpSize) {
      cur_to[col] = scale * cur_from[col];
    }
  }
}

/**
 * query: single row of query
 * key: key_length rows of key
 */
template <bool IsCausal>
__device__ auto get_local_s_m(const float *query, const float *key,
                              float *local_s, uint32_t key_length, uint32_t dim,
                              uint32_t key_step, uint32_t base_q_idx,
                              uint32_t base_key_idx) -> float {
  float local_m = -CUDART_INF_F;
  for (uint32_t thread_iter = 0; thread_iter < kThreadIterPerKV;
       ++thread_iter) {
    auto cur_row_k = (thread_iter * kWarpSize) + (threadIdx.x % kWarpSize);
    if (cur_row_k >= key_length) {
      local_s[thread_iter] = -CUDART_INF_F;
      continue;
    }
    if constexpr (IsCausal) {
      if (base_q_idx < (base_key_idx + cur_row_k)) {
        local_s[thread_iter] = -CUDART_INF_F;
        continue;
      }
    }
    const auto *cur_key = key + (static_cast<size_t>(cur_row_k) * key_step);
    float cur_s = 0.F;
    for (uint32_t i = 0; i < dim; ++i) {
      cur_s += query[i] * cur_key[i];
    }
    local_s[thread_iter] = cur_s;
    local_m = ::max(local_m, cur_s);
  }

  return local_m;
}

/**
 * value: v_length rows of value
 * global_o: size: ceil(dim / kWarpSize)
 */
__device__ auto set_global_m_l_o(const float *value, const float2 &local_m_l,
                                 const float *local_s, float &global_m,
                                 float &global_l, float *global_o,
                                 uint32_t v_length, uint32_t dim,
                                 uint32_t value_step) {
  auto old_l_o_scale = ::expf(global_m - local_m_l.x);
  auto cur_l = (old_l_o_scale * global_l) + local_m_l.y;
  for (uint32_t o_idx = 0, cur_d = threadIdx.x % kWarpSize; cur_d < dim;
       ++o_idx, cur_d += kWarpSize) {
    global_o[o_idx] = old_l_o_scale * global_o[o_idx] * global_l / cur_l;
  }

  for (uint32_t thread_iter = 0; thread_iter < kThreadIterPerKV;
       ++thread_iter) {
    for (int32_t idx = 0; idx < kWarpSize; ++idx) {
      auto cur_row_v = (thread_iter * kWarpSize) + idx;
      if (cur_row_v < v_length) {
        const auto *cur_value =
            value + (static_cast<size_t>(cur_row_v) * value_step);

        auto cur_s =
            ::cuda::device::warp_shuffle_idx(local_s[thread_iter], idx).data;
        for (uint32_t o_idx = 0, cur_d = threadIdx.x % kWarpSize; cur_d < dim;
             ++o_idx, cur_d += kWarpSize) {
          global_o[o_idx] += (cur_s * cur_value[cur_d]) / cur_l;
        }
      }
    }
  }

  global_m = local_m_l.x;
  global_l = cur_l;
}

template <uint32_t OutPerThread, bool IsCausal>
__global__ void flash_attn_kernel(const float *query, const float *key,
                                  const float *value, float *dst,
                                  uint32_t q_head, uint32_t kv_head,
                                  uint32_t q_length, uint32_t kv_length,
                                  uint32_t dim, uint32_t kv_end) {
  __shared__ typename ::cub::WarpReduce<float>::TempStorage
      temp_storage[kWarpNumPerBlock];
  float global_m[kWarpIterPerTileQ];
  float global_l[kWarpIterPerTileQ];
  float global_o[kWarpIterPerTileQ * OutPerThread];
  for (uint32_t i = 0; i < kWarpIterPerTileQ; ++i) {
    global_m[i] = -CUDART_INF_F;
    global_l[i] = 0.F;
    for (uint32_t j = 0; j < OutPerThread; ++j) {
      global_o[(i * OutPerThread) + j] = 0.F;
    }
  }

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  extern __shared__ float buffer[];
  auto padded_dim = GET_PADDED_DIM(dim);
  // float key_buffer[kTileKV][padded_dim]
  auto *key_buffer = (&buffer[0]);
  // float value_buffer[kTileKV][padded_dim]
  auto *value_buffer = key_buffer + (static_cast<size_t>(kTileKV) * padded_dim);
  // float q_buffer[kTileQ][padded_dim]
  auto *query_buffer =
      value_buffer + (static_cast<size_t>(kTileKV) * padded_dim);
  auto q_row_num = ::min(kTileQ, q_length - (blockIdx.x * kTileQ));
  fetch_data(query + get_shift(blockIdx.z, blockIdx.y, blockIdx.x * kTileQ,
                               q_head, q_length, dim),
             query_buffer, q_row_num, dim, dim, padded_dim,
             ::rsqrtf(static_cast<float>(dim)));

  float local_s[kThreadIterPerKV];
  for (uint32_t kv_block_idx = 0,
                kv_block_num = CalBlockNum(kv_length, kTileKV);
       kv_block_idx < kv_block_num; ++kv_block_idx) {
    auto kv_shift = get_shift(blockIdx.z, blockIdx.y / (q_head / kv_head),
                              kv_block_idx * kTileKV, kv_head, kv_end, dim);
    auto kv_row_num = ::min(kTileKV, kv_length - (kv_block_idx * kTileKV));
    fetch_data(key + kv_shift, key_buffer, kv_row_num, dim, dim, padded_dim);
    fetch_data(value + kv_shift, value_buffer, kv_row_num, dim, dim,
               padded_dim);
    __syncthreads();

    for (uint32_t warp_iter = 0; warp_iter * kWarpNumPerBlock < q_row_num;
         ++warp_iter) {
      auto query_buffer_row =
          (warp_iter * kWarpNumPerBlock) + (threadIdx.x / kWarpSize);
      float local_m = get_local_s_m<IsCausal>(
          query_buffer + (static_cast<size_t>(query_buffer_row) * padded_dim),
          key_buffer, &local_s[0], kv_row_num, dim, padded_dim,
          (blockIdx.x * kTileQ) + query_buffer_row, (kv_block_idx * kTileKV));

      local_m = ::cub::WarpReduce<float>(temp_storage[threadIdx.x / kWarpSize])
                    .Max(local_m);
      local_m = ::cuda::device::warp_shuffle_idx(local_m, 0);
      local_m = ::max(local_m, global_m[warp_iter]);
      float local_l{};
      for (float &cur_s : local_s) {
        cur_s = ::exp(cur_s - local_m);
        local_l += cur_s;
      }
      local_l = ::cub::WarpReduce<float>(temp_storage[threadIdx.x / kWarpSize])
                    .Sum(local_l);
      local_l = ::cuda::device::warp_shuffle_idx(local_l, 0);

      set_global_m_l_o(value_buffer, ::make_float2(local_m, local_l),
                       &local_s[0], global_m[warp_iter], global_l[warp_iter],
                       &global_o[warp_iter * OutPerThread], kv_row_num, dim,
                       padded_dim);
      __syncthreads();
    }
  }

  for (uint32_t warp_iter = 0; warp_iter < kWarpIterPerTileQ; ++warp_iter) {
    auto cur_row_q = (blockIdx.x * kTileQ) + (warp_iter * kWarpNumPerBlock) +
                     (threadIdx.x / kWarpSize);
    if (cur_row_q < q_length) {
      auto *dst_ptr = dst + get_shift(blockIdx.z, blockIdx.y, cur_row_q, q_head,
                                      q_length, dim);
      for (uint32_t i = 0; i < OutPerThread; ++i) {
        auto cur_col = (i * kWarpSize) + (threadIdx.x % kWarpSize);
        if (cur_col < dim) {
          dst_ptr[cur_col] = global_o[(warp_iter * OutPerThread) + i];
        }
      }
    }
  }
}
} // namespace

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define CALL_KERNEL(OutPerThread, IsCausal)                                    \
  cudaFuncSetAttribute(                                                        \
      flash_attn_kernel<OutPerThread, IsCausal>,                               \
      cudaFuncAttributeMaxDynamicSharedMemorySize,                             \
      static_cast<int32_t>(                                                    \
          CudaDeviceInfos::SharedMemPerBlockOptin(context.id)));               \
  flash_attn_kernel<OutPerThread, IsCausal>                                    \
      <<<dim3{CalBlockNum(q_length, kTileQ), q_head, batch},                   \
         kThreadNumPerBlock, share_mem_size, context.stream>>>(                \
          query, key, value, dst, q_head, kv_head, q_length, kv_length, dim,   \
          kv_end);

void flash_attn(const float *query, const float *key, const float *value,
                float *dst, uint32_t batch, uint32_t q_head, uint32_t kv_head,
                uint32_t q_length, uint32_t kv_length, uint32_t dim,
                uint32_t kv_end, AttentionType attn_type) {
  TINY_LLM_CHECK(InvalidArgumentError, batch > 0);
  TINY_LLM_CHECK(InvalidArgumentError, q_length > 0);
  TINY_LLM_CHECK(InvalidArgumentError, kv_length > 0);
  TINY_LLM_CHECK(InvalidArgumentError, kv_end > 0);
  TINY_LLM_CHECK(InvalidArgumentError, dim > 0);
  TINY_LLM_CHECK(InvalidArgumentError, q_head > 0);
  TINY_LLM_CHECK(InvalidArgumentError, kv_head > 0);
  TINY_LLM_CHECK(InvalidArgumentError, q_head % kv_head == 0);

  auto context = ThreadCudaContexts::GetContext();
  size_t share_mem_size =
      sizeof(float) * ((2 * kTileKV) + kTileQ) * GET_PADDED_DIM(dim);
  if (share_mem_size > CudaDeviceInfos::SharedMemPerBlockOptin(context.id)) {
    TINY_LLM_THROW_ERROR(
        RuntimeError,
        "The requested shared memory ({}) exceeds the allocatable limit ({}).",
        share_mem_size, CudaDeviceInfos::SharedMemPerBlockOptin(context.id));
  }

  switch (attn_type) {
  case AttentionType::kNoMask: {
    if (dim <= 64) {
      CALL_KERNEL(2, false);
      return;
    }
    if (dim <= 128) {
      CALL_KERNEL(4, false);
      return;
    }
    if (dim <= 256) {
      CALL_KERNEL(8, false);
      return;
    }
    break;
  }
  case AttentionType::kCausalMask: {
    if (dim <= 64) {
      CALL_KERNEL(2, true);
      return;
    }
    if (dim <= 128) {
      CALL_KERNEL(4, true);
      return;
    }
    if (dim <= 256) {
      CALL_KERNEL(8, true);
      return;
    }
    break;
  }
  default:
    break;
  }

  TINY_LLM_THROW_ERROR(NotImplementedError,
                       "The condition `dim={}` is not implemented.", dim);
}

namespace {
constexpr uint32_t kPageSize = 32;

__device__ auto get_shift(uint32_t head_idx, uint32_t seq_idx, uint32_t seq_len,
                          uint32_t head_dim) -> size_t {
  return (static_cast<size_t>(head_idx) * seq_len + seq_idx) * head_dim;
}

/**
 * @brief
 *
 * @param page_pool [num_page, num_head, 32, head_dim]
 * @param to [row_num, to_step]
 * @param page_ids
 * @param start_page_id
 * @param row_num
 * @param head_idx
 * @param num_head
 * @param head_dim
 * @param to_step
 */
__device__ void fetch_page_data(const float *page_pool, float *to,
                                const uint32_t *page_ids,
                                uint32_t start_page_id, uint32_t row_num,
                                uint32_t head_idx, uint32_t num_head,
                                uint32_t head_dim, uint32_t to_step) {
  while (row_num > 0) {
    auto cur_row_num = ::min(32, row_num);
    const auto *base_from_ptr =
        page_pool +
        ((static_cast<size_t>(page_ids[start_page_id]) * num_head + head_idx) *
         32 * head_dim);

    for (uint32_t row = threadIdx.x / kWarpSize; row < cur_row_num;
         row += kWarpNumPerBlock) {
      const auto *cur_from =
          base_from_ptr + (static_cast<size_t>(row) * head_dim);
      auto *cur_to = to + (static_cast<size_t>(row) * to_step);
      for (uint32_t col = threadIdx.x % kWarpSize; col < head_dim;
           col += kWarpSize) {
        cur_to[col] = cur_from[col];
      }
    }

    to += static_cast<size_t>(cur_row_num) * to_step;
    ++start_page_id;
    row_num -= cur_row_num;
  }
}

template <uint32_t OutPerThread, bool IsCausal>
__global__ void flash_attn_paged_kernel(
    const float *query, const float *key_pool, const float *value_pool,
    float *dst, const uint32_t *q_separator, const uint32_t *block_table,
    const uint32_t *kv_lens, uint32_t kv_head, uint32_t head_dim,
    uint32_t max_blocks) {
  __shared__ typename ::cub::WarpReduce<float>::TempStorage
      temp_storage[kWarpNumPerBlock];
  float global_m[kWarpIterPerTileQ];
  float global_l[kWarpIterPerTileQ];
  float global_o[kWarpIterPerTileQ * OutPerThread];
  for (uint32_t i = 0; i < kWarpIterPerTileQ; ++i) {
    global_m[i] = -CUDART_INF_F;
    global_l[i] = 0.F;
    for (uint32_t j = 0; j < OutPerThread; ++j) {
      global_o[(i * OutPerThread) + j] = 0.F;
    }
  }

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables,readability-redundant-declaration)
  extern __shared__ float buffer[];
  auto padded_dim = GET_PADDED_DIM(head_dim);
  // float key_buffer[kTileKV][padded_dim]
  auto *key_buffer = (&buffer[0]);
  // float value_buffer[kTileKV][padded_dim]
  auto *value_buffer = key_buffer + (static_cast<size_t>(kTileKV) * padded_dim);
  // float q_buffer[kTileQ][padded_dim]
  auto *query_buffer =
      value_buffer + (static_cast<size_t>(kTileKV) * padded_dim);

  auto total_queries = q_separator[gridDim.z];
  auto q_length = q_separator[blockIdx.z + 1] - q_separator[blockIdx.z];
  if (blockIdx.x * kTileQ >= q_length) {
    return;
  }
  auto q_row_num = ::min(kTileQ, q_length - (blockIdx.x * kTileQ));
  fetch_data(query + get_shift(blockIdx.y,
                               q_separator[blockIdx.z] + (blockIdx.x * kTileQ),
                               total_queries, head_dim),
             query_buffer, q_row_num, head_dim, head_dim, padded_dim,
             ::rsqrtf(static_cast<float>(head_dim)));

  float local_s[kThreadIterPerKV];
  for (uint32_t kv_block_idx = 0,
                kv_block_num = CalBlockNum(kv_lens[blockIdx.z], kTileKV);
       kv_block_idx < kv_block_num; ++kv_block_idx) {
    auto kv_row_num =
        ::min(kTileKV, kv_lens[blockIdx.z] - (kv_block_idx * kTileKV));
    fetch_page_data(
        key_pool, key_buffer,
        block_table + (static_cast<size_t>(blockIdx.z) * max_blocks),
        kv_block_idx * kThreadIterPerKV, kv_row_num,
        blockIdx.y / (gridDim.y / kv_head), kv_head, head_dim, padded_dim);
    fetch_page_data(
        value_pool, value_buffer,
        block_table + (static_cast<size_t>(blockIdx.z) * max_blocks),
        kv_block_idx * kThreadIterPerKV, kv_row_num,
        blockIdx.y / (gridDim.y / kv_head), kv_head, head_dim, padded_dim);
    __syncthreads();

    for (uint32_t warp_iter = 0; warp_iter * kWarpNumPerBlock < q_row_num;
         ++warp_iter) {
      auto query_buffer_row =
          (warp_iter * kWarpNumPerBlock) + (threadIdx.x / kWarpSize);
      float local_m = get_local_s_m<IsCausal>(
          query_buffer + (static_cast<size_t>(query_buffer_row) * padded_dim),
          key_buffer, &local_s[0], kv_row_num, head_dim, padded_dim,
          (blockIdx.x * kTileQ) + query_buffer_row, (kv_block_idx * kTileKV));
      local_m = ::cub::WarpReduce<float>(temp_storage[threadIdx.x / kWarpSize])
                    .Max(local_m);
      local_m = ::cuda::device::warp_shuffle_idx(local_m, 0);
      local_m = ::max(local_m, global_m[warp_iter]);
      float local_l{};
      for (float &cur_s : local_s) {
        cur_s = ::exp(cur_s - local_m);
        local_l += cur_s;
      }
      local_l = ::cub::WarpReduce<float>(temp_storage[threadIdx.x / kWarpSize])
                    .Sum(local_l);
      local_l = ::cuda::device::warp_shuffle_idx(local_l, 0);

      set_global_m_l_o(value_buffer, ::make_float2(local_m, local_l),
                       &local_s[0], global_m[warp_iter], global_l[warp_iter],
                       &global_o[warp_iter * OutPerThread], kv_row_num,
                       head_dim, padded_dim);
      __syncthreads();
    }
  }

  auto cur_row_q_limit = q_separator[blockIdx.z + 1];
  for (uint32_t warp_iter = 0; warp_iter < kWarpIterPerTileQ; ++warp_iter) {
    auto cur_row_q = q_separator[blockIdx.z] + (blockIdx.x * kTileQ) +
                     (warp_iter * kWarpNumPerBlock) + (threadIdx.x / kWarpSize);
    if (cur_row_q < cur_row_q_limit) {
      auto *dst_ptr =
          dst + get_shift(blockIdx.y, cur_row_q, total_queries, head_dim);
      for (uint32_t i = 0; i < OutPerThread; ++i) {
        auto cur_col = (i * kWarpSize) + (threadIdx.x % kWarpSize);
        if (cur_col < head_dim) {
          dst_ptr[cur_col] = global_o[(warp_iter * OutPerThread) + i];
        }
      }
    }
  }
}
} // namespace

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define CALL_PAGED_KERNEL(OutPerThread, IsCausal)                              \
  cudaFuncSetAttribute(                                                        \
      flash_attn_paged_kernel<OutPerThread, IsCausal>,                         \
      cudaFuncAttributeMaxDynamicSharedMemorySize,                             \
      static_cast<int32_t>(                                                    \
          CudaDeviceInfos::SharedMemPerBlockOptin(context.id)));               \
  flash_attn_paged_kernel<OutPerThread, IsCausal>                              \
      <<<dim3{CalBlockNum(max_q_len, kTileQ), q_head, num_requests},           \
         kThreadNumPerBlock, share_mem_size, context.stream>>>(                \
          query, key_pool, value_pool, dst, q_separator, block_table, kv_lens, \
          kv_head, head_dim, max_blocks);

void flash_attn_paged(const float *query, const float *key_pool,
                      const float *value_pool, float *dst,
                      const uint32_t *q_separator, const uint32_t *block_table,
                      const uint32_t *kv_lens, uint32_t num_requests,
                      uint32_t q_head, uint32_t kv_head, uint32_t head_dim,
                      uint32_t max_blocks, uint32_t page_size,
                      uint32_t max_q_len, AttentionType attn_type) {
  TINY_LLM_CHECK(InvalidArgumentError, num_requests > 0);
  TINY_LLM_CHECK(InvalidArgumentError, q_head > 0);
  TINY_LLM_CHECK(InvalidArgumentError, kv_head > 0);
  TINY_LLM_CHECK(InvalidArgumentError, q_head % kv_head == 0);
  TINY_LLM_CHECK(InvalidArgumentError, head_dim > 0);
  TINY_LLM_CHECK(InvalidArgumentError, max_blocks > 0);
  TINY_LLM_CHECK(InvalidArgumentError, page_size == kPageSize);
  TINY_LLM_CHECK(InvalidArgumentError, max_q_len > 0);

  auto context = ThreadCudaContexts::GetContext();
  size_t share_mem_size =
      sizeof(float) * ((2 * kTileKV) + kTileQ) * GET_PADDED_DIM(head_dim);
  if (share_mem_size > CudaDeviceInfos::SharedMemPerBlockOptin(context.id)) {
    TINY_LLM_THROW_ERROR(
        RuntimeError,
        "The requested shared memory ({}) exceeds the allocatable limit ({}).",
        share_mem_size, CudaDeviceInfos::SharedMemPerBlockOptin(context.id));
  }

  switch (attn_type) {
  case AttentionType::kNoMask: {
    if (head_dim <= 64) {
      CALL_PAGED_KERNEL(2, false);
      return;
    }
    if (head_dim <= 128) {
      CALL_PAGED_KERNEL(4, false);
      return;
    }
    if (head_dim <= 256) {
      CALL_PAGED_KERNEL(8, false);
      return;
    }
    break;
  }
  case AttentionType::kCausalMask: {
    if (head_dim <= 64) {
      CALL_PAGED_KERNEL(2, true);
      return;
    }
    if (head_dim <= 128) {
      CALL_PAGED_KERNEL(4, true);
      return;
    }
    if (head_dim <= 256) {
      CALL_PAGED_KERNEL(8, true);
      return;
    }
    break;
  }
  default:
    break;
  }

  TINY_LLM_THROW_ERROR(NotImplementedError,
                       "The condition `dim={}` is not implemented.", head_dim);
}
// NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index)
} // namespace tiny_llm::cuda
