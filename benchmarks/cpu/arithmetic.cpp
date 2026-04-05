#include "tiny_llm/ops/cpu/arithmetic.hpp"
#include "benchmark/benchmark.h"
#include "tiny_llm/tensor/tensor.hpp"

namespace tiny_llm {
namespace {
void bm_cpu_div(benchmark::State &state) {
  Tensor t({.type = DeviceType::kCpu, .id = 0}, DataType::kFloat32,
           {state.range(0), state.range(0)}, true);

  for (auto _ : state) {
    cpu::arithmetic(t.data<float>(), 2.F, t.data<float>(),
                    static_cast<size_t>(state.range(0) * state.range(0)),
                    ArithmeticType::kDiv);
  }
}
} // namespace

BENCHMARK(bm_cpu_div)
    ->Ranges({{1, 9192}})
    ->RangeMultiplier(8)
    ->Unit(benchmark::kMillisecond);
} // namespace tiny_llm
