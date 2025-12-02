#include "tiny_llm/device_managers/cuda_allocator.hpp"
#include "buffer.hpp"
#include "tiny_llm/common/log_and_excepts.hpp"
#include "tiny_llm/device_managers/cuda_context.hpp"
#include "tiny_llm/device_managers/cuda_guards.hpp"

namespace tiny_llm {
namespace {
class CudaDeleter : public IDeleter {
public:
  CudaDeleter(int32_t dev_id, cudaStream_t stream)
      : dev_id_(dev_id), stream_(stream) {}

  TINY_LLM_DELETE_COPY_MOVE(CudaDeleter);

  ~CudaDeleter() override = default;

  void cleanup(void *ptr) override {
    if (ptr != nullptr) {
      CudaDeviceSwitchGuard guard(dev_id_);
      TINY_LLM_CUDA_WARN(cudaFreeAsync(ptr, stream_));
    }
  }

private:
  int32_t dev_id_{};
  cudaStream_t stream_{};
};

class CudaHostDeleter : public IDeleter {
public:
  CudaHostDeleter() = default;

  TINY_LLM_DELETE_COPY_MOVE(CudaHostDeleter);

  ~CudaHostDeleter() override = default;

  void cleanup(void *ptr) override {
    if (ptr != nullptr) {
      TINY_LLM_CUDA_WARN(cudaFreeHost(ptr));
    }
  }
};
} // namespace

auto CudaAllocator::Allocate(std::size_t size) -> Buffer {
  auto cuda_context = ThreadCudaContexts::GetContext();
  CudaDeviceSwitchGuard guard(cuda_context.id);

  void *ptr{};
  TINY_LLM_CUDA_CHECK(cudaMallocAsync(&ptr, size, cuda_context.stream));

  return {ptr,
          size,
          {.type = DeviceType::kCuda, .id = cuda_context.id},
          std::make_unique<CudaDeleter>(cuda_context.id, cuda_context.stream)};
}

auto CudaHostAllocator::Allocate(std::size_t size) -> Buffer {
  void *ptr{};
  TINY_LLM_CUDA_CHECK(cudaMallocHost(&ptr, size));

  return {ptr,
          size,
          {.type = DeviceType::kCuda, .id = -1},
          std::make_unique<CudaHostDeleter>()};
}
} // namespace tiny_llm
