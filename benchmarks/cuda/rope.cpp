#include "tiny_llm/ops/cuda/rope.hpp"
#include "benchmark/benchmark.h"
#include "tiny_llm/device_managers/cuda/cuda_context.hpp"
#include "tiny_llm/tensor/tensor.hpp"

namespace tiny_llm {
namespace {
void bm_rope(benchmark::State &state) {
  Tensor cos({.type = DeviceType::kCuda, .id = 0}, DataType::kFloat32,
             {state.range(0), state.range(1) / 2}, true);
  Tensor sin({.type = DeviceType::kCuda, .id = 0}, DataType::kFloat32,
             {state.range(0), state.range(1) / 2}, true);
  ThreadCudaContexts::Synchronize();

  for (auto _ : state) {
    cuda::rope(cos.data<float>(), sin.data<float>(), state.range(0),
               state.range(1), 10000.);
    ThreadCudaContexts::Synchronize();
  }
}
} // namespace

BENCHMARK(bm_rope)
    ->Args({{256, 192}})
    ->Args({{192, 256}})
    ->Args({{2048, 1024}})
    ->Args({{1024, 2048}})
    ->Unit(benchmark::kMillisecond);
} // namespace tiny_llm
