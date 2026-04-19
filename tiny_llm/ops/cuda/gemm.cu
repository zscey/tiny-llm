#include <cstddef>

#include "tiny_llm/common/log_and_excepts.hpp"
#include "tiny_llm/ops/cuda/gemm.hpp"

namespace tiny_llm::cuda {
// NOLINTBEGIN(bugprone-easily-swappable-parameters,cppcoreguidelines-pro-bounds-constant-array-index)
namespace {
template <typename T>
concept GemmConfig =
    requires {
      { T::kBlockSize } -> std::convertible_to<uint32_t>;
      { T::kThreadNumX } -> std::convertible_to<uint32_t>;
      { T::kThreadNumY } -> std::convertible_to<uint32_t>;
      { T::kTileX } -> std::convertible_to<uint32_t>;
      { T::kTileY } -> std::convertible_to<uint32_t>;
    } && (T::kBlockSize % 16 == 0) &&
    ((T::kThreadNumX * T::kThreadNumY) % T::kBlockSize == 0);

struct NormalGemmConfig {
  static constexpr uint32_t kBlockSize = 16;
  static constexpr uint32_t kThreadNumX = 16;
  static constexpr uint32_t kThreadNumY = 16;
  static constexpr uint32_t kTileX = 8;
  static constexpr uint32_t kTileY = 8;
};
static_assert(GemmConfig<NormalGemmConfig>);

template <typename T>
concept Indexer = requires(const T &t, uint32_t row, uint32_t col) {
  { t(row, col) } -> std::same_as<size_t>;
};

struct NDIndexer {
  uint32_t stride;
  __device__ __forceinline__ auto operator()(uint32_t row, uint32_t col) const
      -> size_t {
    return (static_cast<size_t>(row) * stride) + col;
  }
};
static_assert(Indexer<NDIndexer>);

struct LTIndexer {
  uint32_t q;
  uint32_t q_start;
  uint32_t q_end;
  uint32_t out_h;
  uint32_t out_d;
  __device__ __forceinline__ auto operator()(uint32_t row, uint32_t col) const
      -> size_t {
    auto cur_b = row / q;
    auto cur_q = q_start + (row % q);
    auto cur_out_h = col / out_d;
    auto cur_out_d = col % out_d;

    return (((((static_cast<size_t>(cur_b) * out_h) + cur_out_h) * q_end) +
             cur_q) *
            out_d) +
           cur_out_d;
  }
};
static_assert(Indexer<LTIndexer>);

struct TLIndexer {
  uint32_t h;
  uint32_t q;
  uint32_t d;
  __device__ __forceinline__ auto operator()(uint32_t row, uint32_t col) const
      -> size_t {
    auto cur_b = row / q;
    auto cur_q = row % q;
    auto cur_h = col / d;
    auto cur_d = col % d;

    return (((((static_cast<size_t>(cur_b) * h) + cur_h) * q) + cur_q) * d) +
           cur_d;
  }
};
static_assert(Indexer<TLIndexer>);

struct SLIndexer {
  uint32_t q;
  uint32_t q_start;
  uint32_t q_end;
  uint32_t d;
  __device__ __forceinline__ auto operator()(uint32_t row, uint32_t col) const
      -> size_t {
    auto cur_b = row / q;
    auto cur_q = q_start + (row % q);

    return (((cur_b * q_end) + cur_q) * d) + col;
  }
};
static_assert(Indexer<SLIndexer>);

template <GemmConfig Config, Indexer FetchIndexer>
__device__ void
// NOLINTNEXTLINE(readability-non-const-parameter)
fetch_input(const float *input, float *input_buffer, uint32_t col_idx,
            uint32_t m, uint32_t d, FetchIndexer fetch_indexer) {
  constexpr auto kBufferStep =
      Config::kThreadNumX * Config::kThreadNumY / Config::kBlockSize;
  constexpr auto kBlockDim = Config::kTileY * Config::kThreadNumY;
  auto buffer_shift = (threadIdx.y * Config::kThreadNumX) + threadIdx.x;
  for (auto buffer_row = buffer_shift / Config::kBlockSize,
            buffer_col = buffer_shift % Config::kBlockSize;
       buffer_row < kBlockDim; buffer_row += kBufferStep) {
    auto &cur_elem =
        input_buffer[(buffer_row * Config::kBlockSize) + buffer_col];
    cur_elem = 0.F;

    auto data_row = (blockIdx.y * kBlockDim) + buffer_row;
    auto data_col = (col_idx * Config::kBlockSize) + buffer_col;
    if (data_row < m && data_col < d) {
      cur_elem = input[fetch_indexer(data_row, data_col)];
    }
  }
}

template <GemmConfig Config, Indexer FetchIndexer>
__device__ __noinline__ void
// NOLINTNEXTLINE(readability-non-const-parameter)
fetch_weight(const float *weight, float *weight_buffer, uint32_t col_idx,
             uint32_t n, uint32_t d, FetchIndexer fetch_indexer) {
  constexpr auto kBufferStep =
      Config::kThreadNumX * Config::kThreadNumY / Config::kBlockSize;
  constexpr auto kBlockDim = Config::kTileX * Config::kThreadNumX;
  auto buffer_shift = (threadIdx.y * Config::kThreadNumX) + threadIdx.x;
  for (auto buffer_row = buffer_shift / Config::kBlockSize,
            buffer_col = buffer_shift % Config::kBlockSize;
       buffer_row < kBlockDim; buffer_row += kBufferStep) {
    auto &cur_elem = weight_buffer[(buffer_row * Config::kBlockSize) +
                                   buffer_row + buffer_col];
    cur_elem = 0.F;

    auto data_row = (blockIdx.x * kBlockDim) + buffer_row;
    auto data_col = (col_idx * Config::kBlockSize) + buffer_col;
    if (data_row < n && data_col < d) {
      cur_elem = weight[fetch_indexer(data_row, data_col)];
    }
  }
}

template <GemmConfig Config, Indexer WriteIndexer>
// NOLINTNEXTLINE(readability-non-const-parameter)
__device__ void write_dst(const float *res, float *dst, uint32_t m, uint32_t n,
                          WriteIndexer write_indexer) {
  for (uint32_t ty = 0; ty < Config::kTileY; ++ty) {
    auto dst_row = (blockIdx.y * Config::kTileY * Config::kThreadNumY) +
                   (ty * Config::kThreadNumY) + threadIdx.y;
    for (uint32_t tx = 0; tx < Config::kTileX; ++tx) {
      auto dst_col = (blockIdx.x * Config::kTileX * Config::kThreadNumX) +
                     (tx * Config::kThreadNumX) + threadIdx.x;
      if (dst_row < m && dst_col < n) {
        dst[write_indexer(dst_row, dst_col)] = res[(ty * Config::kTileX) + tx];
      }
    }
  }
}

template <GemmConfig Config, Indexer InputIndexer, Indexer WeightIndexer,
          Indexer DstIndexer>
__global__ void gemm_no_bias_kernel(const float *input, const float *weight,
                                    float *dst, uint32_t m, uint32_t d,
                                    uint32_t n, InputIndexer input_indexer,
                                    WeightIndexer weight_indexer,
                                    DstIndexer dst_indexer) {
  constexpr uint32_t kBlockDimY = Config::kTileY * Config::kThreadNumY;
  constexpr uint32_t kBlockDimX = Config::kTileX * Config::kThreadNumX;
  __shared__ float input_buffer[kBlockDimY][Config::kBlockSize];
  // The advantages of pad 1 outweigh the disadvantages
  __shared__ float weight_buffer[kBlockDimX][Config::kBlockSize + 1];
  float res[Config::kTileY][Config::kTileX];
  float input_regs[Config::kTileY];
  float weight_regs[Config::kTileX];
  for (auto &re : res) {
    for (float &r : re) {
      r = 0.F;
    }
  }

  auto block_num = CalBlockNum(d, Config::kBlockSize);
  for (uint32_t i = 0; i < block_num; ++i) {
    fetch_input<Config, InputIndexer>(input, &input_buffer[0][0], i, m, d,
                                      input_indexer);
    fetch_weight<Config, WeightIndexer>(weight, &weight_buffer[0][0], i, n, d,
                                        weight_indexer);

    __syncthreads();

    for (uint32_t k = 0; k < Config::kBlockSize; ++k) {
      for (uint32_t ty = 0; ty < Config::kTileY; ++ty) {
        input_regs[ty] =
            input_buffer[threadIdx.y + (ty * Config::kThreadNumY)][k];
      }
      for (uint32_t tx = 0; tx < Config::kTileX; ++tx) {
        weight_regs[tx] =
            weight_buffer[threadIdx.x + (tx * Config::kThreadNumX)][k];
      }
      for (uint32_t ty = 0; ty < Config::kTileY; ++ty) {
        for (uint32_t tx = 0; tx < Config::kTileX; ++tx) {
          res[ty][tx] += input_regs[ty] * weight_regs[tx];
        }
      }
    }

    __syncthreads();
  }

  write_dst<Config, DstIndexer>(&res[0][0], dst, m, n, dst_indexer);
}

constexpr uint32_t kBlockSize = 16;
constexpr uint32_t kTileX = 8;
constexpr uint32_t kTileY = 8;

__device__ void fetch_input(const float *input, float *input_buffer,
                            uint32_t col_block_idx, uint32_t m, uint32_t d) {
  auto buffer_col = (col_block_idx * kBlockSize) + threadIdx.x;
  for (uint32_t ty = 0; ty < kTileY; ++ty) {
    auto *buffer_data =
        input_buffer + ((threadIdx.y * ((kBlockSize * kTileY) + 1)) +
                        (ty * kBlockSize) + threadIdx.x);
    *buffer_data = 0.F;
    auto input_row =
        (blockIdx.y * kBlockSize * kTileY) + (ty * kBlockSize) + threadIdx.y;
    if (buffer_col < d && input_row < m) {
      *buffer_data = input[(input_row * d) + buffer_col];
    }
  }
}

__device__ void fetch_weight(const float *weight, float *weight_buffer,
                             uint32_t col_block_idx, uint32_t n, uint32_t d) {
  auto buffer_col = (col_block_idx * kBlockSize) + threadIdx.x;
  for (uint32_t tx = 0; tx < kTileX; ++tx) {
    auto *buffer_data =
        weight_buffer + ((threadIdx.y * ((kBlockSize * kTileX) + 1)) +
                         (tx * kBlockSize) + threadIdx.x);
    *buffer_data = 0.F;
    auto weight_row =
        (blockIdx.x * kBlockSize * kTileX) + (tx * kBlockSize) + threadIdx.y;
    if (buffer_col < d && weight_row < n) {
      *buffer_data = weight[(weight_row * d) + buffer_col];
    }
  }
}

__device__ void write_dst(const float *res, float *dst, uint32_t m,
                          uint32_t n) {
  for (uint32_t ty = 0; ty < kTileY; ++ty) {
    auto dst_row =
        (blockIdx.y * kBlockSize * kTileY) + (ty * kBlockSize) + threadIdx.y;
    for (uint32_t tx = 0; tx < kTileX; ++tx) {
      auto dst_col =
          (blockIdx.x * kBlockSize * kTileX) + (tx * kBlockSize) + threadIdx.x;
      if (dst_row < m && dst_col < n) {
        dst[(dst_row * n) + dst_col] = res[(ty * kTileX) + tx];
      }
    }
  }
}

__global__ void gemm_row_major_no_bias_kernel(const float *input,
                                              const float *weight, float *dst,
                                              uint32_t m, uint32_t d,
                                              uint32_t n) {
  __shared__ float input_buffer[kBlockSize][(kBlockSize * kTileY) + 1];
  __shared__ float weight_buffer[kBlockSize][(kBlockSize * kTileX) + 1];
  float res[kTileY][kTileX];
  float input_regs[kTileY];
  float weight_regs[kTileX];
  for (auto &re : res) {
    for (float &r : re) {
      r = 0.F;
    }
  }

  auto block_num = CalBlockNum(d, kBlockSize);
  for (uint32_t i = 0; i < block_num; ++i) {
    fetch_input(input, &input_buffer[0][0], i, m, d);
    fetch_weight(weight, &weight_buffer[0][0], i, n, d);
    __syncthreads();

    for (uint32_t k = 0; k < kBlockSize; ++k) {
      for (uint32_t ty = 0; ty < kTileY; ++ty) {
        input_regs[ty] = input_buffer[threadIdx.y][(ty * kBlockSize) + k];
      }
      for (uint32_t tx = 0; tx < kTileX; ++tx) {
        weight_regs[tx] = weight_buffer[threadIdx.x][(tx * kBlockSize) + k];
      }

      for (uint32_t ty = 0; ty < kTileY; ++ty) {
        for (uint32_t tx = 0; tx < kTileX; ++tx) {
          res[ty][tx] += input_regs[ty] * weight_regs[tx];
        }
      }
    }
    __syncthreads();
  }

  write_dst(&res[0][0], dst, m, n);
}

__device__ void write_dst_lt(const float *res, float *dst, uint32_t b,
                             uint32_t q, uint32_t out_head, uint32_t out_d,
                             uint32_t q_start, uint32_t q_end) {
  for (uint32_t ty = 0; ty < kTileY; ++ty) {
    auto dst_row =
        (blockIdx.y * kBlockSize * kTileY) + (ty * kBlockSize) + threadIdx.y;
    auto cur_b = dst_row / q;
    auto cur_q = q_start + (dst_row % q);
    for (uint32_t tx = 0; tx < kTileX; ++tx) {
      auto dst_col =
          (blockIdx.x * kBlockSize * kTileX) + (tx * kBlockSize) + threadIdx.x;
      auto cur_h = dst_col / out_d;
      auto cur_d = dst_col % out_d;
      if (cur_b < b && cur_q < q_end && cur_h < out_head && cur_d < out_d) {
        dst[(((((cur_b * out_head) + cur_h) * q_end) + cur_q) * out_d) +
            cur_d] = res[(ty * kTileX) + tx];
      }
    }
  }
}

__global__ void
gemm_row_major_lt_no_bias_kernel(const float *input, const float *weight,
                                 float *dst, uint32_t b, uint32_t q, uint32_t d,
                                 uint32_t out_head, uint32_t out_d,
                                 uint32_t q_start, uint32_t q_end) {
  __shared__ float input_buffer[kBlockSize][(kBlockSize * kTileY) + 1];
  __shared__ float weight_buffer[kBlockSize][(kBlockSize * kTileX) + 1];
  float res[kTileY][kTileX];
  float input_regs[kTileY];
  float weight_regs[kTileX];
  for (auto &re : res) {
    for (float &r : re) {
      r = 0.F;
    }
  }

  auto block_num = CalBlockNum(d, kBlockSize);
  for (uint32_t i = 0; i < block_num; ++i) {
    fetch_input(input, &input_buffer[0][0], i, (b * q), d);
    fetch_weight(weight, &weight_buffer[0][0], i, (out_head * out_d), d);
    __syncthreads();

    for (uint32_t k = 0; k < kBlockSize; ++k) {
      for (uint32_t ty = 0; ty < kTileY; ++ty) {
        input_regs[ty] = input_buffer[threadIdx.y][(ty * kBlockSize) + k];
      }
      for (uint32_t tx = 0; tx < kTileX; ++tx) {
        weight_regs[tx] = weight_buffer[threadIdx.x][(tx * kBlockSize) + k];
      }

      for (uint32_t ty = 0; ty < kTileY; ++ty) {
        for (uint32_t tx = 0; tx < kTileX; ++tx) {
          res[ty][tx] += input_regs[ty] * weight_regs[tx];
        }
      }
    }
    __syncthreads();
  }

  write_dst_lt(&res[0][0], dst, b, q, out_head, out_d, q_start, q_end);
}

__device__ void fetch_input_tl(const float *input, float *input_buffer,
                               uint32_t col_block_idx, uint32_t b, uint32_t h,
                               uint32_t q, uint32_t d) {
  auto buffer_col = (col_block_idx * kBlockSize) + threadIdx.x;
  for (uint32_t ty = 0; ty < kTileY; ++ty) {
    auto *buffer_data =
        input_buffer + ((threadIdx.y * ((kBlockSize * kTileY) + 1)) +
                        (ty * kBlockSize) + threadIdx.x);
    *buffer_data = 0.F;
    auto input_row =
        (blockIdx.y * kBlockSize * kTileY) + (ty * kBlockSize) + threadIdx.y;
    auto cur_b = input_row / q;
    auto cur_q = input_row % q;
    auto cur_h = buffer_col / d;
    auto cur_d = buffer_col % d;
    if (cur_b < b && cur_h < h && cur_q < q && cur_d < d) {
      *buffer_data = input[((((cur_b * h) + cur_h) * q + cur_q) * d) + cur_d];
    }
  }
}

__global__ void gemm_row_major_tl_no_bias_kernel(const float *input,
                                                 const float *weight,
                                                 float *dst, uint32_t b,
                                                 uint32_t h, uint32_t q,
                                                 uint32_t d, uint32_t out_d) {
  __shared__ float input_buffer[kBlockSize][(kBlockSize * kTileY) + 1];
  __shared__ float weight_buffer[kBlockSize][(kBlockSize * kTileX) + 1];
  float res[kTileY][kTileX];
  float input_regs[kTileY];
  float weight_regs[kTileX];
  for (auto &re : res) {
    for (float &r : re) {
      r = 0.F;
    }
  }

  auto block_num = CalBlockNum(h * d, kBlockSize);
  for (uint32_t i = 0; i < block_num; ++i) {
    fetch_input_tl(input, &input_buffer[0][0], i, b, h, q, d);
    fetch_weight(weight, &weight_buffer[0][0], i, out_d, h * d);
    __syncthreads();

    for (uint32_t k = 0; k < kBlockSize; ++k) {
      for (uint32_t ty = 0; ty < kTileY; ++ty) {
        input_regs[ty] = input_buffer[threadIdx.y][(ty * kBlockSize) + k];
      }
      for (uint32_t tx = 0; tx < kTileX; ++tx) {
        weight_regs[tx] = weight_buffer[threadIdx.x][(tx * kBlockSize) + k];
      }

      for (uint32_t ty = 0; ty < kTileY; ++ty) {
        for (uint32_t tx = 0; tx < kTileX; ++tx) {
          res[ty][tx] += input_regs[ty] * weight_regs[tx];
        }
      }
    }
    __syncthreads();
  }

  write_dst(&res[0][0], dst, b * q, out_d);
}

__device__ void fetch_input_sl(const float *input, float *input_buffer,
                               uint32_t col_block_idx, uint32_t b, uint32_t q,
                               uint32_t d, uint32_t q_start, uint32_t q_end) {
  auto buffer_col = (col_block_idx * kBlockSize) + threadIdx.x;
  for (uint32_t ty = 0; ty < kTileY; ++ty) {
    auto *buffer_data =
        input_buffer + ((threadIdx.y * ((kBlockSize * kTileY) + 1)) +
                        (ty * kBlockSize) + threadIdx.x);
    *buffer_data = 0.F;
    auto input_row =
        (blockIdx.y * kBlockSize * kTileY) + (ty * kBlockSize) + threadIdx.y;
    auto cur_b = input_row / q;
    auto cur_q = q_start + (input_row % q);
    if (cur_b < b && cur_q < q_end && buffer_col < d) {
      *buffer_data = input[(((cur_b * q_end) + cur_q) * d) + buffer_col];
    }
  }
}

__global__ void
gemm_row_major_sl_no_bias_kernel(const float *input, const float *weight,
                                 float *dst, uint32_t b, uint32_t q, uint32_t d,
                                 uint32_t n, uint32_t q_start, uint32_t q_end) {
  __shared__ float input_buffer[kBlockSize][(kBlockSize * kTileY) + 1];
  __shared__ float weight_buffer[kBlockSize][(kBlockSize * kTileX) + 1];
  float res[kTileY][kTileX];
  float input_regs[kTileY];
  float weight_regs[kTileX];
  for (auto &re : res) {
    for (float &r : re) {
      r = 0.F;
    }
  }

  auto block_num = CalBlockNum(d, kBlockSize);
  for (uint32_t i = 0; i < block_num; ++i) {
    fetch_input_sl(input, &input_buffer[0][0], i, b, q, d, q_start, q_end);
    fetch_weight(weight, &weight_buffer[0][0], i, n, d);
    __syncthreads();

    for (uint32_t k = 0; k < kBlockSize; ++k) {
      for (uint32_t ty = 0; ty < kTileY; ++ty) {
        input_regs[ty] = input_buffer[threadIdx.y][(ty * kBlockSize) + k];
      }
      for (uint32_t tx = 0; tx < kTileX; ++tx) {
        weight_regs[tx] = weight_buffer[threadIdx.x][(tx * kBlockSize) + k];
      }

      for (uint32_t ty = 0; ty < kTileY; ++ty) {
        for (uint32_t tx = 0; tx < kTileX; ++tx) {
          res[ty][tx] += input_regs[ty] * weight_regs[tx];
        }
      }
    }
    __syncthreads();
  }

  write_dst(&res[0][0], dst, b * q, n);
}
} // namespace

void gemm_row_major(const float *input, const float *weight, const float *bias,
                    float *dst, uint32_t m, uint32_t d, uint32_t n) {
  TINY_LLM_CHECK(bias == nullptr);
  if (m == 0 || d == 0 || n == 0) {
    return;
  }

  gemm_row_major_no_bias_kernel<<<dim3{CalBlockNum(n, kBlockSize * kTileX),
                                       CalBlockNum(m, kBlockSize * kTileY)},
                                  dim3{kBlockSize, kBlockSize}, 0,
                                  ThreadCudaContexts::GetContext().stream>>>(
      input, weight, dst, m, d, n);
}

void gemm_row_major_lt(const float *input, const float *weight,
                       const float *bias, float *dst, uint32_t b, uint32_t q,
                       uint32_t d, uint32_t out_head, uint32_t out_d,
                       uint32_t q_start, uint32_t q_end) {
  TINY_LLM_CHECK(bias == nullptr);
  TINY_LLM_CHECK(q_start + q <= q_end);
  if (b == 0 || q == 0 || d == 0 || out_head == 0 || out_d == 0) {
    return;
  }

  gemm_row_major_lt_no_bias_kernel<<<
      dim3{CalBlockNum((out_head * out_d), kBlockSize * kTileX),
           CalBlockNum((b * q), kBlockSize * kTileY)},
      dim3{kBlockSize, kBlockSize}, 0,
      ThreadCudaContexts::GetContext().stream>>>(
      input, weight, dst, b, q, d, out_head, out_d, q_start, q_end);
}

void gemm_row_major_tl(const float *input, const float *weight,
                       const float *bias, float *dst, uint32_t b, uint32_t h,
                       uint32_t q, uint32_t d, uint32_t out_d) {
  TINY_LLM_CHECK(bias == nullptr);
  if (b == 0 || h == 0 || q == 0 || d == 0 || out_d == 0) {
    return;
  }

  gemm_row_major_tl_no_bias_kernel<<<
      dim3{CalBlockNum(out_d, kBlockSize * kTileX),
           CalBlockNum((b * q), kBlockSize * kTileY)},
      dim3{kBlockSize, kBlockSize}, 0,
      ThreadCudaContexts::GetContext().stream>>>(input, weight, dst, b, h, q, d,
                                                 out_d);
}

void gemm_row_major_sl(const float *input, const float *weight,
                       const float *bias, float *dst, uint32_t b, uint32_t q,
                       uint32_t d, uint32_t n, uint32_t q_start,
                       uint32_t q_end) {
  TINY_LLM_CHECK(bias == nullptr);
  TINY_LLM_CHECK(q_start + q <= q_end);
  if (b == 0 || q == 0 || d == 0 || n == 0) {
    return;
  }

  gemm_row_major_sl_no_bias_kernel<<<dim3{CalBlockNum(n, kBlockSize * kTileX),
                                          CalBlockNum((b * q),
                                                      kBlockSize * kTileY)},
                                     dim3{kBlockSize, kBlockSize}, 0,
                                     ThreadCudaContexts::GetContext().stream>>>(
      input, weight, dst, b, q, d, n, q_start, q_end);
}
// NOLINTEND(bugprone-easily-swappable-parameters,cppcoreguidelines-pro-bounds-constant-array-index)
} // namespace tiny_llm::cuda
