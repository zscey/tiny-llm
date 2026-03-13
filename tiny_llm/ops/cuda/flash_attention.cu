#include "cccl/cuda/warp"
#include "cub/cub.cuh"
#include "math_constants.h"
#include "tiny_llm/common/log_and_excepts.hpp"
#include "tiny_llm/device_managers/cuda/cuda_device_infos.hpp"
#include "tiny_llm/ops/cuda/flash_attention.hpp"

namespace tiny_llm::cuda {
// NOLINTBEGIN(bugprone-easily-swappable-parameters,cppcoreguidelines-pro-bounds-constant-array-index)
namespace {
constexpr uint32_t kWarpSize = 32;
constexpr uint32_t kWarpNumPerBlock = 32;
constexpr uint32_t kThreadNumPerBlock = kWarpNumPerBlock * kWarpSize;
constexpr uint32_t kWarpIterPerTileQ = 2;
constexpr uint32_t kTileQ = kWarpIterPerTileQ * kWarpNumPerBlock;
constexpr uint32_t kThreadIterPerKV = 2;
constexpr uint32_t kTileKV = kThreadIterPerKV * kWarpSize;

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define GET_PADDED_DIM(dim)                                                    \
  ((((dim) + kWarpSize - 1) / kWarpSize * kWarpSize) + 1)

__device__ auto get_shift(uint32_t batch_idx, uint32_t head_idx,
                          uint32_t seq_idx, uint32_t seq_len, uint32_t head_num,
                          uint32_t dim) -> size_t {
  return ((((static_cast<size_t>(batch_idx) * seq_len) + seq_idx) * head_num) +
          head_idx) *
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
__device__ auto get_local_s_m_l(const float *query, const float *key,
                                float *local_s, uint32_t key_length,
                                uint32_t dim, uint32_t key_step) -> float2 {
  float local_m = -CUDART_INF_F;
  float local_l{};
  for (uint32_t thread_iter = 0; thread_iter < kThreadIterPerKV;
       ++thread_iter) {
    auto cur_row_k = (thread_iter * kWarpSize) + (threadIdx.x % kWarpSize);
    if (cur_row_k < key_length) {
      const auto *cur_key = key + (static_cast<size_t>(cur_row_k) * key_step);
      auto &cur_s = local_s[thread_iter];
      cur_s = 0.F;
      for (uint32_t i = 0; i < dim; ++i) {
        cur_s += query[i] * cur_key[i];
      }
      local_m = ::max(local_m, cur_s);
      local_l += ::expf(cur_s);
    }
  }

  return ::make_float2(local_m, local_l);
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
  auto cur_m = ::max(local_m_l.x, global_m);
  auto old_l_o_scale = ::expf(global_m - cur_m);
  auto cur_l = (old_l_o_scale * global_l) + (::expf(-cur_m) * local_m_l.y);
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
        auto soft_max_scale = ::expf(cur_s - cur_m);
        for (uint32_t o_idx = 0, cur_d = threadIdx.x % kWarpSize; cur_d < dim;
             ++o_idx, cur_d += kWarpSize) {
          global_o[o_idx] += (soft_max_scale * cur_value[cur_d]) / cur_l;
        }
      }
    }
  }

  global_m = cur_m;
  global_l = cur_l;
}

template <uint32_t OutPerThread>
__global__ void
flash_attn_kernel(const float *query, const float *key, const float *value,
                  float *dst, uint32_t q_length, uint32_t kv_length,
                  uint32_t dim, uint32_t q_head, uint32_t kv_head) {
  __shared__ typename ::cub::WarpReduce<float>::TempStorage
      temp_storage[kWarpNumPerBlock];
  float global_m[kWarpIterPerTileQ];
  float global_l[kWarpIterPerTileQ];
  float global_o[kWarpIterPerTileQ * OutPerThread];
  for (uint32_t i = 0; i < kWarpIterPerTileQ; ++i) {
    global_m[i] = -CUDART_INF_F;
    global_l[i] = CUDART_MIN_DENORM_F;
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
                               q_length, q_head, dim),
             query_buffer, q_row_num, dim, q_head * dim, padded_dim,
             ::rsqrtf(static_cast<float>(dim)));

  float local_s[kThreadIterPerKV];
  for (uint32_t kv_block_idx = 0,
                kv_block_num = CalBlockNum(kv_length, kTileKV);
       kv_block_idx < kv_block_num; ++kv_block_idx) {
    auto kv_shift = get_shift(blockIdx.z, blockIdx.y / (q_head / kv_head),
                              kv_block_idx * kTileKV, kv_length, kv_head, dim);
    auto kv_row_num = ::min(kTileKV, kv_length - (kv_block_idx * kTileKV));
    fetch_data(key + kv_shift, key_buffer, kv_row_num, dim, kv_head * dim,
               padded_dim);
    fetch_data(value + kv_shift, value_buffer, kv_row_num, dim, kv_head * dim,
               padded_dim);
    __syncthreads();

    for (uint32_t warp_iter = 0; warp_iter * kWarpNumPerBlock < q_row_num;
         ++warp_iter) {
      auto local_m_l = get_local_s_m_l(
          query_buffer + (static_cast<size_t>((warp_iter * kWarpNumPerBlock) +
                                              (threadIdx.x / kWarpSize)) *
                          padded_dim),
          key_buffer, &local_s[0], kv_row_num, dim, padded_dim);

      local_m_l.x =
          ::cub::WarpReduce<float>(temp_storage[threadIdx.x / kWarpSize])
              .Max(local_m_l.x);
      local_m_l.x = ::cuda::device::warp_shuffle_idx(local_m_l.x, 0);
      local_m_l.y =
          ::cub::WarpReduce<float>(temp_storage[threadIdx.x / kWarpSize])
              .Sum(local_m_l.y);
      local_m_l.y = ::cuda::device::warp_shuffle_idx(local_m_l.y, 0);

      set_global_m_l_o(value_buffer, local_m_l, &local_s[0],
                       global_m[warp_iter], global_l[warp_iter],
                       &global_o[warp_iter * OutPerThread], kv_row_num, dim,
                       padded_dim);
      __syncthreads();
    }
  }

  for (uint32_t warp_iter = 0; warp_iter < kWarpIterPerTileQ; ++warp_iter) {
    auto cur_row_q = (blockIdx.x * kTileQ) + (warp_iter * kWarpNumPerBlock) +
                     (threadIdx.x / kWarpSize);
    if (cur_row_q < q_length) {
      auto *dst_ptr = dst + get_shift(blockIdx.z, blockIdx.y, cur_row_q,
                                      q_length, q_head, dim);
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
#define CALL_KERNEL(OutPerThread)                                              \
  cudaFuncSetAttribute(                                                        \
      flash_attn_kernel<OutPerThread>,                                         \
      cudaFuncAttributeMaxDynamicSharedMemorySize,                             \
      static_cast<int32_t>(                                                    \
          CudaDeviceInfos::SharedMemPerBlockOptin(context.id)));               \
  flash_attn_kernel<OutPerThread>                                              \
      <<<dim3{CalBlockNum(q_length, kTileQ), q_head, batch},                   \
         kThreadNumPerBlock, share_mem_size, context.stream>>>(                \
          query, key, value, dst, q_length, kv_length, dim, q_head, kv_head);

void flash_attn(const float *query, const float *key, const float *value,
                float *dst, uint32_t batch, uint32_t q_length,
                uint32_t kv_length, uint32_t dim, uint32_t q_head,
                uint32_t kv_head) {
  TINY_LLM_CHECK(batch > 0);
  TINY_LLM_CHECK(q_length > 0);
  TINY_LLM_CHECK(kv_length > 0);
  TINY_LLM_CHECK(dim > 0);
  TINY_LLM_CHECK(q_head > 0);
  TINY_LLM_CHECK(kv_head > 0);
  TINY_LLM_CHECK(q_head % kv_head == 0);

  auto context = ThreadCudaContexts::GetContext();
  size_t share_mem_size =
      sizeof(float) * ((2 * kTileKV) + kTileQ) * GET_PADDED_DIM(dim);
  if (share_mem_size > CudaDeviceInfos::SharedMemPerBlockOptin(context.id)) {
    TINY_LLM_THROW_ERROR(
        std::runtime_error,
        "The requested shared memory ({}) exceeds the allocatable limit ({}).",
        share_mem_size, CudaDeviceInfos::SharedMemPerBlockOptin(context.id));
  }
  TINY_LLM_CHECK(share_mem_size);

  if (dim <= 64) {
    CALL_KERNEL(2);
    return;
  }
  if (dim <= 128) {
    CALL_KERNEL(4);
    return;
  }
  if (dim <= 256) {
    CALL_KERNEL(8);
    return;
  }

  TINY_LLM_THROW_ERROR(std::runtime_error,
                       "The condition `dim={}` is not implemented.", dim);
}
// NOLINTEND(bugprone-easily-swappable-parameters,cppcoreguidelines-pro-bounds-constant-array-index)
} // namespace tiny_llm::cuda
