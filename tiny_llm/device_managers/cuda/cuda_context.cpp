#include "tiny_llm/device_managers/cuda/cuda_context.hpp"
#include "tiny_llm/common/checks.hpp"
#include "tiny_llm/common/cuda_checks.hpp"
#include "tiny_llm/common/exception.hpp"
#include "tiny_llm/device_managers/cuda/cuda_device_guard.hpp"
#include <array>
#include <atomic>
#include <stack>
#include <vector>

namespace tiny_llm {
// ========================== CudaContextAllocator ==========================
namespace {
constexpr size_t kStreamsPerDevice = 32;

struct alignas(64) ContextResources {
  std::atomic_size_t pos{0};
  std::array<cudaStream_t, kStreamsPerDevice> stream_pool{};
};

void check_dev_id(int32_t dev_id, int32_t dev_num) {
  TINY_LLM_CHECK(InvalidArgumentError, dev_id >= 0);
  TINY_LLM_CHECK(InvalidArgumentError, dev_id < dev_num);
}
} // namespace

class CudaContextAllocator::Impl {
public:
  std::vector<std::unique_ptr<ContextResources>> resources;
};

auto CudaContextAllocator::Instance() -> CudaContextAllocator & {
  static CudaContextAllocator cuda_stream_allocator;
  return cuda_stream_allocator;
}

CudaContextAllocator::CudaContextAllocator() : impl_(std::make_unique<Impl>()) {
  int32_t dev_num{};
  TINY_LLM_CUDA_CHECK(CudaError, cudaGetDeviceCount(&dev_num));

  CudaDeviceSwitchGuard guard(-1);

  impl_->resources.reserve(dev_num);
  for (int32_t dev_id = 0; dev_id < dev_num; ++dev_id) {
    auto &cur_resource =
        impl_->resources.emplace_back(std::make_unique<ContextResources>());
    cur_resource->pos = 0;

    TINY_LLM_CUDA_CHECK(CudaError, cudaSetDevice(dev_id));
    for (auto &stream : cur_resource->stream_pool) {
      TINY_LLM_CUDA_CHECK(CudaError, cudaStreamCreate(&stream));
    }
  }
}

CudaContextAllocator::~CudaContextAllocator() noexcept = default;

auto CudaContextAllocator::CreateCudaContext(int32_t dev_id) -> CudaContext {
  if (dev_id < 0) {
    TINY_LLM_CUDA_CHECK(CudaError, cudaGetDevice(&dev_id));
  }
  auto &instance = Instance();
  check_dev_id(dev_id, static_cast<int32_t>(instance.impl_->resources.size()));

  auto &cur_resource = instance.impl_->resources.at(dev_id);
  auto cur_pos = cur_resource->pos.load(std::memory_order_relaxed);
  auto next_pos = (cur_pos + 1) % kStreamsPerDevice;
  while (!cur_resource->pos.compare_exchange_weak(cur_pos, next_pos,
                                                  std::memory_order_acq_rel,
                                                  std::memory_order_relaxed)) {
    next_pos = (cur_pos + 1) % kStreamsPerDevice;
  };

  return {.stream = cur_resource->stream_pool.at(cur_pos),
          .id = static_cast<DeviceId>(dev_id)};
}

// ========================== ThreadCudaContexts ==========================
class ThreadCudaContexts::Impl {
public:
  std::vector<std::stack<CudaContext>> contexts;
};

ThreadCudaContexts::ThreadCudaContexts() : impl_(std::make_unique<Impl>()) {
  int32_t dev_num{};
  TINY_LLM_CUDA_CHECK(CudaError, cudaGetDeviceCount(&dev_num));

  impl_->contexts.resize(dev_num);
};

auto ThreadCudaContexts::ThreadInstance() -> ThreadCudaContexts & {
  thread_local static ThreadCudaContexts thread_cuda_contexts;
  return thread_cuda_contexts;
}

void ThreadCudaContexts::Push(CudaContext cuda_context) {
  auto &instance = ThreadInstance();
  check_dev_id(cuda_context.id,
               static_cast<int32_t>(instance.impl_->contexts.size()));

  instance.impl_->contexts.at(cuda_context.id).push(cuda_context);
}

void ThreadCudaContexts::Pop(int32_t dev_id) {
  auto &instance = ThreadInstance();
  if (dev_id < 0) {
    TINY_LLM_CUDA_CHECK(CudaError, cudaGetDevice(&dev_id));
  }
  check_dev_id(dev_id, static_cast<int32_t>(instance.impl_->contexts.size()));

  auto &cur_contexts = instance.impl_->contexts.at(dev_id);
  if (!cur_contexts.empty()) {
    cur_contexts.pop();
  }
}

auto ThreadCudaContexts::GetContext() -> CudaContext {
  auto &instance = ThreadInstance();
  int32_t dev_id{};
  TINY_LLM_CUDA_CHECK(CudaError, cudaGetDevice(&dev_id));

  auto &cur_contexts = instance.impl_->contexts.at(dev_id);
  if (cur_contexts.empty()) {
    cur_contexts.push(CudaContextAllocator::CreateCudaContext(dev_id));
  }

  return cur_contexts.top();
}

void ThreadCudaContexts::Synchronize() {
  int32_t dev_id{};
  TINY_LLM_CUDA_CHECK(CudaError, cudaGetDevice(&dev_id));

  auto &cur_contexts = ThreadInstance().impl_->contexts.at(dev_id);
  if (!cur_contexts.empty()) {
    TINY_LLM_CUDA_CHECK(CudaError,
                        cudaStreamSynchronize(cur_contexts.top().stream));
  }
}

void ThreadCudaContexts::SynchronizeDevice() {
  TINY_LLM_CUDA_CHECK(CudaError, cudaDeviceSynchronize());
}

// ========================== ThreadCudaContextsGuard ==========================
ThreadCudaContextsGuard::ThreadCudaContextsGuard(CudaContext cuda_context)
    : context_dev_id_(cuda_context.id) {
  ThreadCudaContexts::Push(cuda_context);
}

ThreadCudaContextsGuard::~ThreadCudaContextsGuard() noexcept {
  try {
    ThreadCudaContexts::Pop(context_dev_id_);
  } catch (std::exception &ex) {
    SPDLOG_WARN(ex.what());
  }
}
} // namespace tiny_llm
