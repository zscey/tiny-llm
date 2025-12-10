#include "tiny_llm/device_managers/cuda/cuda_guards.hpp"
#include "tiny_llm/common/log_and_excepts.hpp"

namespace tiny_llm {
// ========================== CudaDeviceSwitchGuard ==========================
CudaDeviceSwitchGuard::CudaDeviceSwitchGuard(int32_t target_dev) {
  TINY_LLM_CUDA_CHECK(cudaGetDevice(&origin_dev_));
  if (origin_dev_ != target_dev) {
    TINY_LLM_CUDA_CHECK(cudaSetDevice(target_dev));
  }
}

CudaDeviceSwitchGuard::~CudaDeviceSwitchGuard() noexcept {
  int32_t cur_dev{};
  TINY_LLM_CUDA_WARN(cudaGetDevice(&cur_dev));
  if (cur_dev != origin_dev_) {
    TINY_LLM_CUDA_WARN(cudaSetDevice(origin_dev_));
  }
}

// ========================== ThreadCudaContextsGuard ==========================
ThreadCudaContextsGuard::ThreadCudaContextsGuard(CudaContext cuda_context) {
  ThreadCudaContexts::Push(cuda_context);
}

ThreadCudaContextsGuard::~ThreadCudaContextsGuard() noexcept {
  ThreadCudaContexts::Pop();
}

} // namespace tiny_llm
