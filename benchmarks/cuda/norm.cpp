#include "tiny_llm/ops/cuda/norm.hpp"
#include "benchmark/benchmark.h"
#include "tiny_llm/device_managers/cuda/cuda_context.hpp"
#include "tiny_llm/tensor/tensor.hpp"

namespace tiny_llm {
namespace {
void bm_norm(benchmark::State &state) {
  Tensor input({.type = DeviceType::kCuda, .id = 0}, DataType::kFloat32,
               {2, state.range(0), state.range(1)}, true);
  Tensor weight({.type = DeviceType::kCuda, .id = 0}, DataType::kFloat32,
                {state.range(1)}, true);
  Tensor dst({.type = DeviceType::kCuda, .id = 0}, DataType::kFloat32,
             {2, state.range(0), state.range(1)}, true);
  ThreadCudaContexts::Synchronize();

  for (auto _ : state) {
    cuda::rms_norm(input.data<float>(), weight.data<float>(), dst.data<float>(),
                   2 * state.range(0), state.range(1));
    ThreadCudaContexts::Synchronize();
  }
}
} // namespace

BENCHMARK(bm_norm)
    ->Ranges({{256, 4096}, {16, 256}})
    ->RangeMultiplier(4)
    ->Unit(benchmark::kMillisecond);
} // namespace tiny_llm
