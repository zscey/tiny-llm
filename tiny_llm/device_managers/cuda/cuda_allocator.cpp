#include "tiny_llm/device_managers/cuda/cuda_allocator.hpp"
#include "tiny_llm/common/cuda_checks.hpp"
#include "tiny_llm/common/exception.hpp"
#include "tiny_llm/device_managers/cuda/cuda_context.hpp"
#include "tiny_llm/device_managers/cuda/cuda_device_guard.hpp"

namespace tiny_llm {
namespace {
class CudaDeleter : public IDeleter {
public:
  explicit CudaDeleter(CudaContext cuda_context)
      : cuda_context_(cuda_context) {}

  TINY_LLM_DELETE_COPY_MOVE(CudaDeleter);

  ~CudaDeleter() override = default;

  void cleanup(void *ptr) override {
    if (ptr != nullptr) {
      CudaDeviceSwitchGuard guard(cuda_context_.id);
      TINY_LLM_CUDA_WARN(cudaFreeAsync(ptr, cuda_context_.stream));
    }
  }

private:
  CudaContext cuda_context_{};
};

} // namespace

auto CudaAllocator::Allocate(std::size_t size) -> Buffer {
  auto cuda_context = ThreadCudaContexts::GetContext();

  void *ptr{};
  if (size > 0) {
    TINY_LLM_CUDA_CHECK(tiny_llm::CudaError,
                        cudaMallocAsync(&ptr, size, cuda_context.stream));
  }

  return {ptr,
          size,
          {.type = DeviceType::kCuda, .id = cuda_context.id},
          std::make_unique<CudaDeleter>(cuda_context)};
}

namespace {
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

auto CudaHostAllocator::Allocate(std::size_t size) -> Buffer {
  void *ptr{};
  if (size > 0) {
    TINY_LLM_CUDA_CHECK(tiny_llm::CudaError, cudaMallocHost(&ptr, size));
  }

  return {ptr,
          size,
          {.type = DeviceType::kCudaHost, .id = 0},
          std::make_unique<CudaHostDeleter>()};
}
} // namespace tiny_llm
