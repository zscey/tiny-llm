#pragma once

#include "tiny_llm/common/common_macros.hpp"
#include "tiny_llm/device_managers/cuda_context.hpp"
#include <cstdint>

namespace tiny_llm {
/// @brief Set device to `target_dev` when constructing, and reset device to
/// `origin_dev_` when destroying.
class CudaDeviceSwitchGuard {
public:
  explicit CudaDeviceSwitchGuard(int32_t target_dev);

  TINY_LLM_DELETE_COPY_MOVE(CudaDeviceSwitchGuard);

  ~CudaDeviceSwitchGuard() noexcept;

private:
  int32_t origin_dev_{};
};

/// @brief Set `cuda_context` as the cuda context of the current thread when
/// constructing, and reset the original cuda context of the current thread when
/// destroying.
class ThreadCudaContextsGuard {
public:
  explicit ThreadCudaContextsGuard(CudaContext cuda_context);

  TINY_LLM_DELETE_COPY_MOVE(ThreadCudaContextsGuard);

  ~ThreadCudaContextsGuard() noexcept;
};
} // namespace tiny_llm
