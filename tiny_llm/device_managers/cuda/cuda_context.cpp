#include "tiny_llm/device_managers/cuda/cuda_context.hpp"
#include "tiny_llm/common/log_and_excepts.hpp"
#include "tiny_llm/device_managers/cuda/cuda_device_guard.hpp"
#include <array>
#include <atomic>
#include <stack>

namespace tiny_llm {
// ========================== CudaContextAllocator ==========================
class CudaContextAllocator::Impl {
public:
  std::array<std::array<cudaStream_t, CUDA_STREAM_POOL_SIZE>,
             MAX_CUDA_DEVICE_NUM>
      stream_pool{};
  std::array<std::atomic_size_t, MAX_CUDA_DEVICE_NUM> stream_pos{};
};

auto CudaContextAllocator::Instance() -> CudaContextAllocator & {
  static CudaContextAllocator cuda_stream_allocator;
  return cuda_stream_allocator;
}

CudaContextAllocator::CudaContextAllocator() : impl_(std::make_unique<Impl>()) {
  int32_t dev_num{};
  TINY_LLM_CUDA_CHECK(cudaGetDeviceCount(&dev_num));

  CudaDeviceSwitchGuard guard(-1);

  for (int32_t dev_id = 0, dev_id_end = std::min(dev_num, MAX_CUDA_DEVICE_NUM);
       dev_id < dev_id_end; ++dev_id) {
    impl_->stream_pos.at(dev_id) = 0;

    TINY_LLM_CUDA_CHECK(cudaSetDevice(dev_id));

    auto &streams = impl_->stream_pool.at(dev_id);
    for (auto &stream : streams) {
      TINY_LLM_CUDA_CHECK(cudaStreamCreate(&stream));
    }
  }
}

// CudaContextAllocator::~CudaContextAllocator() noexcept = default;
CudaContextAllocator::~CudaContextAllocator() noexcept {
  int32_t dev_num{};
  TINY_LLM_CUDA_WARN(cudaGetDeviceCount(&dev_num));

  CudaDeviceSwitchGuard guard(-1);

  for (int32_t dev_id = 0, dev_id_end = std::min(dev_num, MAX_CUDA_DEVICE_NUM);
       dev_id < dev_id_end; ++dev_id) {
    TINY_LLM_CUDA_WARN(cudaSetDevice(dev_id));

    auto &streams = impl_->stream_pool.at(dev_id);
    for (auto &stream : streams) {
      TINY_LLM_CUDA_WARN(cudaStreamDestroy(stream));
    }
  }
}

auto CudaContextAllocator::CreateCudaContext(int32_t dev_id) -> CudaContext {
  if (dev_id < 0) {
    TINY_LLM_CUDA_CHECK(cudaGetDevice(&dev_id));
  }
  TINY_LLM_CHECK(dev_id < MAX_CUDA_DEVICE_NUM);

  auto &cuda_stream_allocator = Instance();
  auto &cur_stream_pos = cuda_stream_allocator.impl_->stream_pos.at(dev_id);

  auto cur_pos = cur_stream_pos.load(std::memory_order_relaxed);
  auto next_pos = (cur_pos + 1) % CUDA_STREAM_POOL_SIZE;
  while (!cur_stream_pos.compare_exchange_weak(cur_pos, next_pos,
                                               std::memory_order_acq_rel,
                                               std::memory_order_relaxed)) {
    next_pos = (cur_pos + 1) % CUDA_STREAM_POOL_SIZE;
  };

  return {.stream =
              cuda_stream_allocator.impl_->stream_pool.at(dev_id).at(cur_pos),
          .id = static_cast<DeviceId>(dev_id)};
}

// ========================== ThreadCudaContexts ==========================
class ThreadCudaContexts::Impl {
public:
  std::array<std::stack<CudaContext>, MAX_CUDA_DEVICE_NUM> contexts{};
};

ThreadCudaContexts::ThreadCudaContexts() : impl_(std::make_unique<Impl>()) {};

auto ThreadCudaContexts::ThreadInstance() -> ThreadCudaContexts & {
  thread_local static ThreadCudaContexts thread_cuda_contexts;
  return thread_cuda_contexts;
}

void ThreadCudaContexts::Push(CudaContext cuda_context) {
  TINY_LLM_CHECK(0 <= cuda_context.id);
  TINY_LLM_CHECK(cuda_context.id < MAX_CUDA_DEVICE_NUM);

  ThreadInstance().impl_->contexts.at(cuda_context.id).push(cuda_context);
}

void ThreadCudaContexts::Pop(int32_t dev_id) {
  if (dev_id < 0) {
    TINY_LLM_CUDA_CHECK(cudaGetDevice(&dev_id));
  }
  TINY_LLM_CHECK(dev_id < MAX_CUDA_DEVICE_NUM);

  auto &cur_contexts = ThreadInstance().impl_->contexts.at(dev_id);
  if (!cur_contexts.empty()) {
    cur_contexts.pop();
  }
}

auto ThreadCudaContexts::GetContext() -> CudaContext {
  int32_t dev_id{};
  TINY_LLM_CUDA_CHECK(cudaGetDevice(&dev_id));
  TINY_LLM_CHECK(dev_id < MAX_CUDA_DEVICE_NUM);

  auto &cur_contexts = ThreadInstance().impl_->contexts.at(dev_id);
  if (cur_contexts.empty()) {
    cur_contexts.push(CudaContextAllocator::CreateCudaContext());
  }

  return cur_contexts.top();
}

void ThreadCudaContexts::Synchronize() {
  int32_t dev_id{};
  TINY_LLM_CUDA_CHECK(cudaGetDevice(&dev_id));
  TINY_LLM_CHECK(dev_id < MAX_CUDA_DEVICE_NUM);

  auto &cur_contexts = ThreadInstance().impl_->contexts.at(dev_id);
  if (!cur_contexts.empty()) {
    TINY_LLM_CUDA_CHECK(cudaStreamSynchronize(cur_contexts.top().stream));
  }
}

void ThreadCudaContexts::SynchronizeDevice() {
  TINY_LLM_CUDA_CHECK(cudaDeviceSynchronize());
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
    SPDLOG_ERROR(ex.what());
  }
}
} // namespace tiny_llm
