#include "tiny_llm/ops/cuda/rope.hpp"
#include "benchmark/benchmark.h"
#include "tiny_llm/device_managers/cuda/cuda_context.hpp"
#include "tiny_llm/tensor/tensor.hpp"

namespace tiny_llm {
namespace {
void bm_rope(benchmark::State &state) {
  int64_t max_len = state.range(0);
  int64_t dim = state.range(1);
  Tensor cos({.type = DeviceType::kCuda, .id = 0}, DataType::kFloat32,
             {max_len, dim / 2}, true);
  Tensor sin({.type = DeviceType::kCuda, .id = 0}, DataType::kFloat32,
             {max_len, dim / 2}, true);
  ThreadCudaContexts::Synchronize();

  for (auto _ : state) {
    cuda::rope(cos.data<float>(), sin.data<float>(), max_len, dim, 10000.);
    ThreadCudaContexts::Synchronize();
  }
}

void bm_apply_rope(benchmark::State &state) {
  int64_t b = 2;
  int64_t h = 4;
  int64_t max_len = state.range(0);
  int64_t dim = state.range(1);
  Tensor cos({.type = DeviceType::kCuda, .id = 0}, DataType::kFloat32,
             {max_len, dim / 2}, true);
  Tensor sin({.type = DeviceType::kCuda, .id = 0}, DataType::kFloat32,
             {max_len, dim / 2}, true);
  Tensor pos_ids({.type = DeviceType::kCpu, .id = 0}, DataType::kUint32,
                 {b, max_len}, true);
  auto *pos_ids_ptr = pos_ids.data<uint32_t>();
  for (uint32_t i = 0, i_end = (b * max_len); i < i_end; ++i) {
    pos_ids_ptr[i] = i % max_len;
  }
  pos_ids = pos_ids.to({.type = DeviceType::kCuda, .id = 0});
  Tensor dst({.type = DeviceType::kCuda, .id = 0}, DataType::kFloat32,
             {b, h, max_len, dim}, true);
  ThreadCudaContexts::Synchronize();

  for (auto _ : state) {
    cuda::apply_rope_inplace(cos.data<float>(), sin.data<float>(),
                             pos_ids.data<uint32_t>(), dst.data<float>(), b, h,
                             max_len, dim, 0, max_len);
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

BENCHMARK(bm_apply_rope)
    ->Args({{256, 192}})
    ->Args({{192, 256}})
    ->Args({{2048, 1024}})
    ->Args({{1024, 2048}})
    ->Unit(benchmark::kMillisecond);
} // namespace tiny_llm
