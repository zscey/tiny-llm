#pragma once

#include "cuda_runtime.h"
#include "tiny_llm/common/common_macros.hpp"
#include "tiny_llm/device_managers/buffer.hpp"
#include <array>
#include <atomic>
#include <memory>
#include <stack>

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
  /// @brief Create a cuda context on the current device.
  static auto CreateCudaContext() -> CudaContext;

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
  /// @brief Add a cuda context on the current device; `cuda_context.id` must be
  /// equal to the current device ID.
  static void Push(CudaContext cuda_context);

  /// @brief Remove a cuda context on the current device.
  static void Pop();

  /// @brief Get the cuda context on the current device. This interface always
  /// returns a valid cuda context.
  static auto GetContext() -> CudaContext;

  /// @brief Synchronize with the current cuda context.
  static void Synchronize();

  /// @brief Synchronizer with the current device.
  static void SynchronizeDevice();

private:
  ThreadCudaContexts();

  static auto ThreadInstance() -> ThreadCudaContexts &;

  class Impl;

  std::unique_ptr<Impl> impl_;
};
} // namespace tiny_llm
