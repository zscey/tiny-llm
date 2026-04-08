#include "tiny_llm/device_managers/cuda/cuda_context.hpp"
#include "tiny_llm/common/log_and_excepts.hpp"
#include "tiny_llm/device_managers/cuda/cuda_device_guard.hpp"
#include <array>
#include <atomic>
#include <stack>
#include <vector>

namespace tiny_llm {
// ========================== CudaContextAllocator ==========================
namespace {
constexpr size_t kStreamsPerDevice = 32;

struct alignas(64) PaddedAtomic {
  std::atomic_size_t pos{0};
};
} // namespace

class CudaContextAllocator::Impl {
public:
  std::vector<std::array<cudaStream_t, kStreamsPerDevice>> stream_pool;
  std::unique_ptr<PaddedAtomic[]> stream_pos;
};

auto CudaContextAllocator::Instance() -> CudaContextAllocator & {
  static CudaContextAllocator cuda_stream_allocator;
  return cuda_stream_allocator;
}

CudaContextAllocator::CudaContextAllocator() : impl_(std::make_unique<Impl>()) {
  int32_t dev_num{};
  TINY_LLM_CUDA_CHECK(cudaGetDeviceCount(&dev_num));

  CudaDeviceSwitchGuard guard(-1);

  impl_->stream_pool.resize(dev_num);
  impl_->stream_pos = std::make_unique<PaddedAtomic[]>(dev_num);
  for (int32_t dev_id = 0; dev_id < dev_num; ++dev_id) {
    impl_->stream_pos[dev_id].pos = 0;

    TINY_LLM_CUDA_CHECK(cudaSetDevice(dev_id));
    auto &streams = impl_->stream_pool.at(dev_id);
    for (auto &stream : streams) {
      TINY_LLM_CUDA_CHECK(cudaStreamCreate(&stream));
    }
  }
}

CudaContextAllocator::~CudaContextAllocator() noexcept = default;

auto CudaContextAllocator::CreateCudaContext(int32_t dev_id) -> CudaContext {
  if (dev_id < 0) {
    TINY_LLM_CUDA_CHECK(cudaGetDevice(&dev_id));
  }

  auto &cuda_stream_allocator = Instance();
  auto &cur_stream_pos = cuda_stream_allocator.impl_->stream_pos[dev_id].pos;

  auto cur_pos = cur_stream_pos.load(std::memory_order_relaxed);
  auto next_pos = (cur_pos + 1) % kStreamsPerDevice;
  while (!cur_stream_pos.compare_exchange_weak(cur_pos, next_pos,
                                               std::memory_order_acq_rel,
                                               std::memory_order_relaxed)) {
    next_pos = (cur_pos + 1) % kStreamsPerDevice;
  };

  return {.stream =
              cuda_stream_allocator.impl_->stream_pool.at(dev_id).at(cur_pos),
          .id = static_cast<DeviceId>(dev_id)};
}

// ========================== ThreadCudaContexts ==========================
class ThreadCudaContexts::Impl {
public:
  std::vector<std::stack<CudaContext>> contexts;
};

ThreadCudaContexts::ThreadCudaContexts() : impl_(std::make_unique<Impl>()) {
  int32_t dev_num{};
  TINY_LLM_CUDA_CHECK(cudaGetDeviceCount(&dev_num));

  impl_->contexts.resize(dev_num);
};

auto ThreadCudaContexts::ThreadInstance() -> ThreadCudaContexts & {
  thread_local static ThreadCudaContexts thread_cuda_contexts;
  return thread_cuda_contexts;
}

void ThreadCudaContexts::Push(CudaContext cuda_context) {
  auto &instance = ThreadInstance();
  TINY_LLM_CHECK(0 <= cuda_context.id);
  TINY_LLM_CHECK(static_cast<size_t>(cuda_context.id) <
                 instance.impl_->contexts.size());

  instance.impl_->contexts.at(cuda_context.id).push(cuda_context);
}

void ThreadCudaContexts::Pop(int32_t dev_id) {
  auto &instance = ThreadInstance();
  if (dev_id < 0) {
    TINY_LLM_CUDA_CHECK(cudaGetDevice(&dev_id));
  }

  auto &cur_contexts = instance.impl_->contexts.at(dev_id);
  if (!cur_contexts.empty()) {
    cur_contexts.pop();
  }
}

auto ThreadCudaContexts::GetContext() -> CudaContext {
  auto &instance = ThreadInstance();
  int32_t dev_id{};
  TINY_LLM_CUDA_CHECK(cudaGetDevice(&dev_id));

  auto &cur_contexts = instance.impl_->contexts.at(dev_id);
  if (cur_contexts.empty()) {
    cur_contexts.push(CudaContextAllocator::CreateCudaContext(dev_id));
  }

  return cur_contexts.top();
}

void ThreadCudaContexts::Synchronize() {
  int32_t dev_id{};
  TINY_LLM_CUDA_CHECK(cudaGetDevice(&dev_id));

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
