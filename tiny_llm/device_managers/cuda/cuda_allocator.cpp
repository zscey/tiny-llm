#include "tiny_llm/device_managers/cuda/cuda_allocator.hpp"
#include "tiny_llm/common/checks.hpp"
#include "tiny_llm/common/cuda_checks.hpp"
#include "tiny_llm/common/exception.hpp"
#include "tiny_llm/device_managers/cuda/cuda_context.hpp"
#include "tiny_llm/device_managers/cuda/cuda_device_guard.hpp"
#include <mutex>
#include <unordered_map>

namespace tiny_llm {
namespace {
class PtrInfo {
public:
  static void insert(void *ptr, CudaContext cuda_context) {
    auto &instance = Instance();
    {
      std::lock_guard guard(instance.mtx_);
      instance.ptr_meta_.emplace(ptr, cuda_context);
    }
  }

  static auto remove(void *ptr) -> CudaContext {
    auto &instance = Instance();
    {
      std::lock_guard guard(instance.mtx_);
      auto iter = instance.ptr_meta_.find(ptr);
      if (iter == instance.ptr_meta_.end()) {
        TINY_LLM_THROW_ERROR(RuntimeError,
                             "The current pointer is not managed by tiny_llm.");
      }
      auto res = iter->second;
      instance.ptr_meta_.erase(iter);
      return res;
    }
  }

private:
  std::mutex mtx_;
  std::unordered_map<void *, CudaContext> ptr_meta_;
  PtrInfo() = default;

  static auto Instance() -> PtrInfo & {
    static PtrInfo ptr_info;
    return ptr_info;
  }
};

void cuda_deleter(void *ptr) {
  if (ptr == nullptr) {
    return;
  }

  try {
    auto cuda_context = PtrInfo::remove(ptr);
    CudaDeviceSwitchGuard guard(cuda_context.id);
    TINY_LLM_CUDA_CHECK(CudaError, cudaFreeAsync(ptr, cuda_context.stream));
  } catch (std::exception &ex) {
    SPDLOG_WARN(ex.what());
  }
}
} // namespace

auto CudaAllocator::Allocate(std::size_t size) -> Buffer {
  auto cuda_context = ThreadCudaContexts::GetContext();

  void *ptr{};
  if (size > 0) {
    TINY_LLM_CUDA_CHECK(tiny_llm::CudaError,
                        cudaMallocAsync(&ptr, size, cuda_context.stream));
    PtrInfo::insert(ptr, cuda_context);
  }

  return {ptr,
          size,
          {.type = DeviceType::kCuda, .id = cuda_context.id},
          cuda_deleter};
}

namespace {
void cuda_host_deleter(void *ptr) {
  if (ptr == nullptr) {
    return;
  }
  TINY_LLM_CUDA_WARN(cudaFreeHost(ptr));
}
} // namespace

auto CudaHostAllocator::Allocate(std::size_t size) -> Buffer {
  void *ptr{};
  if (size > 0) {
    TINY_LLM_CUDA_CHECK(tiny_llm::CudaError, cudaMallocHost(&ptr, size));
  }

  return {
      ptr, size, {.type = DeviceType::kCudaHost, .id = 0}, cuda_host_deleter};
}
} // namespace tiny_llm
