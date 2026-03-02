#include "tiny_llm/ops/cuda/embedding.hpp"
#include "benchmark/benchmark.h"
#include "tiny_llm/device_managers/cuda/cuda_context.hpp"
#include "tiny_llm/tensor/tensor.hpp"

namespace tiny_llm {
namespace {
void bm_embedding(benchmark::State &state) {
  Tensor emb_weight({.type = DeviceType::kCuda, .id = 0}, DataType::kFloat32,
                    {state.range(1), state.range(2)}, true);
  Tensor src({.type = DeviceType::kCpu, .id = 0}, DataType::kUint32,
             {state.range(0)}, true);
  auto *src_data = src.data<uint32_t>();
  for (int64_t i = 0; i < state.range(0); ++i) {
    src_data[i] = i % state.range(1);
  }
  src = src.to({.type = DeviceType::kCuda, .id = 0});
  Tensor dst({.type = DeviceType::kCuda, .id = 0}, DataType::kFloat32,
             {state.range(0), state.range(2)}, true);
  ThreadCudaContexts::Synchronize();

  for (auto _ : state) {
    cuda::embedding(src.data<uint32_t>(), src.data<float>(), dst.data<float>(),
                    state.range(2), state.range(0));
    ThreadCudaContexts::Synchronize();
  }
}
} // namespace

BENCHMARK(bm_embedding)
    ->Args({{256, 100, 192}})
    ->Args({{2560, 1000, 1920}})
    ->Unit(benchmark::kMillisecond);
} // namespace tiny_llm
