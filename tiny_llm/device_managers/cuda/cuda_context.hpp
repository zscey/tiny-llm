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

  // add other features below
};

/// @brief `CudaContextAllocator` is used to create `CudaContext`, and it is the
/// owner of the resources used in `CudaContext`.
class CudaContextAllocator {
public:
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
  static void Push(CudaContext cuda_context);

  static void Pop();

  static auto GetContext() -> CudaContext;

  /// @brief Synchronize cur context.
  static void Synchronize();

  /// @brief Synchronizer cur device.
  static void SynchronizeDevice();

private:
  ThreadCudaContexts();

  static auto ThreadInstance() -> ThreadCudaContexts &;

  class Impl;

  std::unique_ptr<Impl> impl_;
};
} // namespace tiny_llm
