#include "tiny_llm/device_managers/cuda/cuda_device_guard.hpp"
#include "cuda_runtime.h"
#include "tiny_llm/common/cuda_checks.hpp"
#include "tiny_llm/common/exception.hpp"

namespace tiny_llm {
CudaDeviceSwitchGuard::CudaDeviceSwitchGuard(int32_t target_dev) {
  TINY_LLM_CUDA_CHECK(CudaError, cudaGetDevice(&origin_dev_));
  if (target_dev >= 0 && origin_dev_ != target_dev) {
    TINY_LLM_CUDA_CHECK(CudaError, cudaSetDevice(target_dev));
  }
}

CudaDeviceSwitchGuard::~CudaDeviceSwitchGuard() noexcept {
  int32_t cur_dev{};
  TINY_LLM_CUDA_WARN(cudaGetDevice(&cur_dev));
  if (cur_dev != origin_dev_) {
    TINY_LLM_CUDA_WARN(cudaSetDevice(origin_dev_));
  }
}

} // namespace tiny_llm
