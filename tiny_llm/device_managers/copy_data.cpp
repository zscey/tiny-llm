#include "tiny_llm/device_managers/copy_data.hpp"
#include "tiny_llm/common/checks.hpp"
#include "tiny_llm/common/exception.hpp"

#ifdef TINY_LLM_COPY_DATA_WITH_CUDA
#include "tiny_llm/common/cuda_checks.hpp"
#include "tiny_llm/device_managers/cuda/cuda_context.hpp"
#endif

namespace tiny_llm {
void copy_data(const void *src_ptr, Device src_device, void *dst_ptr,
               Device dst_device, std::size_t size) {
  if (size == 0) {
    return;
  }

  switch (src_device.type) {
#ifdef TINY_LLM_COPY_DATA_WITH_CUDA
  case DeviceType::kCudaHost:
#endif
  case DeviceType::kCpu: {
    switch (dst_device.type) {
#ifdef TINY_LLM_COPY_DATA_WITH_CUDA
    case DeviceType::kCuda:
      TINY_LLM_CUDA_CHECK(
          tiny_llm::CudaError,
          cudaMemcpyAsync(dst_ptr, src_ptr, size, cudaMemcpyHostToDevice,
                          ThreadCudaContexts::GetContext().stream));
      return;
    case DeviceType::kCudaHost:
#endif
    case DeviceType::kCpu:
      std::memcpy(dst_ptr, src_ptr, size);
      return;
    default:
      break;
    }
  }
#ifdef TINY_LLM_COPY_DATA_WITH_CUDA
  case DeviceType::kCuda: {
    switch (dst_device.type) {
    case DeviceType::kCuda:
      TINY_LLM_CHECK(tiny_llm::NotImplementedError,
                     src_device.id == dst_device.id);
      TINY_LLM_CUDA_CHECK(
          tiny_llm::CudaError,
          cudaMemcpyAsync(dst_ptr, src_ptr, size, cudaMemcpyDeviceToDevice,
                          ThreadCudaContexts::GetContext().stream));
      return;
    case DeviceType::kCudaHost:
    case DeviceType::kCpu:
      TINY_LLM_CUDA_CHECK(
          tiny_llm::CudaError,
          cudaMemcpyAsync(dst_ptr, src_ptr, size, cudaMemcpyDeviceToHost,
                          ThreadCudaContexts::GetContext().stream));
      return;
    default:
      break;
    }
  }
#endif
  default:
    break;
  }

  TINY_LLM_THROW_ERROR(tiny_llm::RuntimeError,
                       "Copy from [{}:{}] to [{}:{}] not implemented.",
                       to_string(src_device.type), src_device.id,
                       to_string(dst_device.type), dst_device.id);
}
} // namespace tiny_llm
