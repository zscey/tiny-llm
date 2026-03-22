#include "tiny_llm/ops/cuda/gemm.hpp"
#include "benchmark/benchmark.h"
#include "tiny_llm/device_managers/cuda/cuda_context.hpp"
#include "tiny_llm/tensor/tensor.hpp"

namespace tiny_llm {
namespace {
void bm_gemm(benchmark::State &state) {
  Tensor input({.type = DeviceType::kCuda, .id = 0}, DataType::kFloat32,
               {state.range(0), state.range(1)}, true);
  Tensor weight({.type = DeviceType::kCuda, .id = 0}, DataType::kFloat32,
                {state.range(2), state.range(1)}, true);
  Tensor dst({.type = DeviceType::kCuda, .id = 0}, DataType::kFloat32,
             {state.range(0), state.range(2)}, true);
  ThreadCudaContexts::Synchronize();

  for (auto _ : state) {
    cuda::gemm_row_major(input.data<float>(), weight.data<float>(), nullptr,
                         dst.data<float>(), state.range(0), state.range(1),
                         state.range(2));
    ThreadCudaContexts::Synchronize();
  }
}

void bm_gemm_lt(benchmark::State &state) {
  int64_t b = 2;
  int64_t q = state.range(0) / b;
  int64_t d = state.range(1);
  int64_t out_head = 4;
  int64_t out_dim = state.range(2) / out_head;
  int64_t q_start = 3;
  int64_t q_end = q + q_start;

  Tensor input({.type = DeviceType::kCuda, .id = 0}, DataType::kFloat32,
               {b, q, d}, true);
  Tensor weight({.type = DeviceType::kCuda, .id = 0}, DataType::kFloat32,
                {out_head, out_dim, d}, true);
  Tensor dst({.type = DeviceType::kCuda, .id = 0}, DataType::kFloat32,
             {b, out_head, q_end, out_dim}, true);
  ThreadCudaContexts::Synchronize();

  for (auto _ : state) {
    cuda::gemm_row_major_lt(input.data<float>(), weight.data<float>(), nullptr,
                            dst.data<float>(), b, q, d, out_head, out_dim,
                            q_start, q_end);
    ThreadCudaContexts::Synchronize();
  }
}

void bm_gemm_tl(benchmark::State &state) {
  int64_t b = 2;
  int64_t h = 4;
  int64_t q = state.range(0) / b;
  int64_t d = state.range(1) / h;
  int64_t out_d = state.range(2);

  Tensor input({.type = DeviceType::kCuda, .id = 0}, DataType::kFloat32,
               {b, h, q, d}, true);
  Tensor weight({.type = DeviceType::kCuda, .id = 0}, DataType::kFloat32,
                {out_d, h * d}, true);
  Tensor dst({.type = DeviceType::kCuda, .id = 0}, DataType::kFloat32,
             {b, out_d, h * d}, true);
  ThreadCudaContexts::Synchronize();

  for (auto _ : state) {
    cuda::gemm_row_major_tl(input.data<float>(), weight.data<float>(), nullptr,
                            dst.data<float>(), b, h, q, d, out_d);
    ThreadCudaContexts::Synchronize();
  }
}
} // namespace

BENCHMARK(bm_gemm)
    ->Args({127, 511, 243})
    ->Args({511, 127, 243})
    ->Args({4095, 8191, 6033})
    ->Args({8191, 4095, 6033})
    ->Args({10245, 10241, 10243})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(bm_gemm_lt)
    ->Args({127, 511, 243})
    ->Args({511, 127, 243})
    ->Args({4095, 8191, 6033})
    ->Args({8191, 4095, 6033})
    ->Args({10245, 10241, 10243})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(bm_gemm_tl)
    ->Args({127, 511, 243})
    ->Args({511, 127, 243})
    ->Args({4095, 8191, 6033})
    ->Args({8191, 4095, 6033})
    ->Args({10245, 10241, 10243})
    ->Unit(benchmark::kMillisecond);
} // namespace tiny_llm
