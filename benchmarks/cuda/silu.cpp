#include "tiny_llm/ops/cuda/silu.hpp"
#include "benchmark/benchmark.h"
#include "tiny_llm/device_managers/cuda/cuda_context.hpp"
#include "tiny_llm/tensor/tensor.hpp"

namespace tiny_llm {
namespace {
void bm_silu(benchmark::State &state) {
  Tensor src({.type = DeviceType::kCuda, .id = 0}, DataType::kFloat32,
             {state.range(0), state.range(0)}, true);
  Tensor dst({.type = DeviceType::kCuda, .id = 0}, DataType::kFloat32,
             {state.range(0), state.range(0)}, true);
  ThreadCudaContexts::Synchronize();

  for (auto _ : state) {
    cuda::silu(src.data<float>(), dst.data<float>(),
               static_cast<size_t>(state.range(0) * state.range(0)));
    ThreadCudaContexts::Synchronize();
  }
}
} // namespace

BENCHMARK(bm_silu)
    ->Ranges({{1, 9192}})
    ->RangeMultiplier(8)
    ->Unit(benchmark::kMillisecond);
} // namespace tiny_llm
