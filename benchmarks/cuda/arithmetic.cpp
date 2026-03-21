#include "tiny_llm/ops/cuda/arithmetic.hpp"
#include "benchmark/benchmark.h"
#include "tiny_llm/device_managers/cuda/cuda_context.hpp"
#include "tiny_llm/tensor/tensor.hpp"

namespace tiny_llm {
namespace {
void bm_add(benchmark::State &state) {
  Tensor t({.type = DeviceType::kCuda, .id = 0}, DataType::kFloat32,
           {state.range(0), state.range(0)}, true);
  ThreadCudaContexts::Synchronize();

  for (auto _ : state) {
    cuda::arithmetic(t.data<float>(), t.data<float>(), t.data<float>(),
                     static_cast<size_t>(state.range(0) * state.range(0)),
                     cuda::ArithmeticType::kAdd);
    ThreadCudaContexts::Synchronize();
  }
}
} // namespace

BENCHMARK(bm_add)
    ->Ranges({{1, 9192}})
    ->RangeMultiplier(8)
    ->Unit(benchmark::kMillisecond);
} // namespace tiny_llm
