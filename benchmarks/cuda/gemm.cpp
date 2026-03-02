#include "tiny_llm/ops/cuda/gemm.hpp"
#include "benchmark/benchmark.h"
#include "tiny_llm/device_managers/cuda/cuda_context.hpp"
#include "tiny_llm/tensor/tensor.hpp"

namespace tiny_llm {
namespace {
void bm_gemm_plain(benchmark::State &state) {
  Tensor input({.type = DeviceType::kCuda, .id = 0}, DataType::kFloat32,
               {state.range(0), state.range(1)}, true);
  Tensor weight({.type = DeviceType::kCuda, .id = 0}, DataType::kFloat32,
                {state.range(2), state.range(1)}, true);
  Tensor dst({.type = DeviceType::kCuda, .id = 0}, DataType::kFloat32,
             {state.range(0), state.range(2)}, true);
  ThreadCudaContexts::Synchronize();

  for (auto _ : state) {
    cuda::gemm_row_major_plain(input.data<float>(), weight.data<float>(),
                               nullptr, dst.data<float>(), state.range(0),
                               state.range(1), state.range(2));
    ThreadCudaContexts::Synchronize();
  }
}

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

} // namespace

BENCHMARK(bm_gemm_plain)
    ->Args({127, 511, 243})
    ->Args({511, 127, 243})
    ->Args({4095, 8191, 6033})
    ->Args({8191, 4095, 6033})
    ->Args({10245, 10241, 10243})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(bm_gemm)
    ->Args({127, 511, 243})
    ->Args({511, 127, 243})
    ->Args({4095, 8191, 6033})
    ->Args({8191, 4095, 6033})
    ->Args({10245, 10241, 10243})
    ->Unit(benchmark::kMillisecond);

} // namespace tiny_llm
