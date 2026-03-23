#include "tiny_llm/ops/cuda/flash_attention.hpp"
#include "benchmark/benchmark.h"
#include "tiny_llm/device_managers/cuda/cuda_context.hpp"
#include "tiny_llm/tensor/tensor.hpp"

namespace tiny_llm {
namespace {
void bm_flash_attn_no_mask(benchmark::State &state) {
  int64_t batch = 2;
  int64_t q_head = 8;
  int64_t kv_head = 4;
  int64_t seq_len = state.range(0);
  uint32_t dim = 64;

  Tensor query({.type = DeviceType::kCuda, .id = 0}, DataType::kFloat32,
               {batch, q_head, seq_len, dim}, true);
  Tensor key({.type = DeviceType::kCuda, .id = 0}, DataType::kFloat32,
             {batch, kv_head, seq_len, dim}, true);
  Tensor value({.type = DeviceType::kCuda, .id = 0}, DataType::kFloat32,
               {batch, kv_head, seq_len, dim}, true);
  Tensor dst({.type = DeviceType::kCuda, .id = 0}, DataType::kFloat32,
             {batch, q_head, seq_len, dim}, true);
  ThreadCudaContexts::Synchronize();

  for (auto _ : state) {
    cuda::flash_attn(query.data<float>(), key.data<float>(),
                     value.data<float>(), dst.data<float>(), batch, q_head,
                     kv_head, seq_len, seq_len, dim, seq_len,
                     cuda::AttentionType::kNoMask);
    ThreadCudaContexts::Synchronize();
  }
}

void bm_flash_attn_causal_mask(benchmark::State &state) {
  int64_t batch = 2;
  int64_t q_head = 8;
  int64_t kv_head = 4;
  int64_t seq_len = state.range(0);
  uint32_t dim = 64;

  Tensor query({.type = DeviceType::kCuda, .id = 0}, DataType::kFloat32,
               {batch, q_head, seq_len, dim}, true);
  Tensor key({.type = DeviceType::kCuda, .id = 0}, DataType::kFloat32,
             {batch, kv_head, seq_len, dim}, true);
  Tensor value({.type = DeviceType::kCuda, .id = 0}, DataType::kFloat32,
               {batch, kv_head, seq_len, dim}, true);
  Tensor dst({.type = DeviceType::kCuda, .id = 0}, DataType::kFloat32,
             {batch, q_head, seq_len, dim}, true);
  ThreadCudaContexts::Synchronize();

  for (auto _ : state) {
    cuda::flash_attn(query.data<float>(), key.data<float>(),
                     value.data<float>(), dst.data<float>(), batch, q_head,
                     kv_head, seq_len, seq_len, dim, seq_len,
                     cuda::AttentionType::kCausalMask);
    ThreadCudaContexts::Synchronize();
  }
}
} // namespace

BENCHMARK(bm_flash_attn_no_mask)
    ->Ranges({{1, 9192}})
    ->RangeMultiplier(8)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(bm_flash_attn_causal_mask)
    ->Ranges({{1, 9192}})
    ->RangeMultiplier(8)
    ->Unit(benchmark::kMillisecond);
} // namespace tiny_llm
