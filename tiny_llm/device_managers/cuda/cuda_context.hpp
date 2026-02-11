#pragma once

#include "cuda_runtime.h"
#include "tiny_llm/common/common_macros.hpp"
#include "tiny_llm/device_managers/buffer.hpp"
#include <memory>

namespace tiny_llm {
/// @brief A set of cuda features. `CudaContext` doesn't manage any resources,
/// therefore it can be freely copied and moved.
struct CudaContext {
  cudaStream_t stream{cudaStreamPerThread};
  DeviceId id{0};

  // Add other features below
};

/// @brief `CudaContextAllocator` is used to create `CudaContext`, and it is the
/// owner of the resources used in `CudaContext`.
class CudaContextAllocator {
public:
  /**
   * @brief Create a `CudaContext` object on the specified device.
   *
   * @param dev_id The specified device ID. If negative, it represents the
   * current device.
   * @return CudaContext
   */
  static auto CreateCudaContext(int32_t dev_id = -1) -> CudaContext;

  TINY_LLM_DELETE_COPY_MOVE(CudaContextAllocator);

  ~CudaContextAllocator() noexcept;

private:
  static auto Instance() -> CudaContextAllocator &;

  CudaContextAllocator();

  class Impl;

  std::unique_ptr<Impl> impl_;
};

/// @brief A class that manages the current thread context.
class ThreadCudaContexts {
public:
  /// @brief Add a `CudaContext` to the device specified by `cuda_context.id`.
  static void Push(CudaContext cuda_context);

  /**
   * @brief Remove a `CudaContext` on the specified device.
   *
   * @param dev_id The specified device ID. If negative, it represents the
   * current device.
   */
  static void Pop(int32_t dev_id = -1);

  /// @brief Get the `CudaContext` on the current device. This interface always
  /// returns a valid `CudaContext`.
  static auto GetContext() -> CudaContext;

  /// @brief Synchronize with the current `CudaContext`.
  static void Synchronize();

  /// @brief Synchronizer with the current device.
  static void SynchronizeDevice();

private:
  ThreadCudaContexts();

  static auto ThreadInstance() -> ThreadCudaContexts &;

  class Impl;

  std::unique_ptr<Impl> impl_;
};

/// @brief During construction, `cuda_context` is pushed to the current thread.
/// And during destruction, a `CudaContext` with `cuda_context.id` is popped.
class ThreadCudaContextsGuard {
public:
  explicit ThreadCudaContextsGuard(CudaContext cuda_context);

  TINY_LLM_DELETE_COPY_MOVE(ThreadCudaContextsGuard);

  ~ThreadCudaContextsGuard() noexcept;

private:
  int32_t context_dev_id_{};
};
} // namespace tiny_llm
