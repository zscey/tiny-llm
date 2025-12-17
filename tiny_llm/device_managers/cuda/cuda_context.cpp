#include "tiny_llm/device_managers/cuda/cuda_context.hpp"
#include "tiny_llm/common/log_and_excepts.hpp"

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

  int32_t cur_dev{};
  TINY_LLM_CUDA_CHECK(cudaGetDevice(&cur_dev));

  for (int32_t dev_id = 0, dev_id_end = std::min(dev_num, MAX_CUDA_DEVICE_NUM);
       dev_id < dev_id_end; ++dev_id) {
    impl_->stream_pos.at(dev_id) = 0;

    TINY_LLM_CUDA_CHECK(cudaSetDevice(dev_id));

    auto &streams = impl_->stream_pool.at(dev_id);
    for (size_t stream_id = 0; stream_id < CUDA_STREAM_POOL_SIZE; ++stream_id) {
      TINY_LLM_CUDA_CHECK(cudaStreamCreate(&streams.at(stream_id)));
    }
  }

  TINY_LLM_CUDA_CHECK(cudaSetDevice(cur_dev));
}

CudaContextAllocator::~CudaContextAllocator() noexcept {
  int32_t dev_num{};
  TINY_LLM_CUDA_WARN(cudaGetDeviceCount(&dev_num));

  int32_t cur_dev{};
  TINY_LLM_CUDA_WARN(cudaGetDevice(&cur_dev));

  for (int32_t dev_id = 0, dev_id_end = std::min(dev_num, MAX_CUDA_DEVICE_NUM);
       dev_id < dev_id_end; ++dev_id) {
    TINY_LLM_CUDA_WARN(cudaSetDevice(dev_id));

    auto &streams = impl_->stream_pool.at(dev_id);
    for (size_t stream_id = 0; stream_id < CUDA_STREAM_POOL_SIZE; ++stream_id) {
      TINY_LLM_CUDA_WARN(cudaStreamDestroy(streams.at(stream_id)));
    }
  }

  TINY_LLM_CUDA_WARN(cudaSetDevice(cur_dev));
}

auto CudaContextAllocator::CreateCudaContext() -> CudaContext {
  int32_t cur_dev{};
  TINY_LLM_CUDA_CHECK(cudaGetDevice(&cur_dev));

  auto &cuda_stream_allocator = Instance();
  auto &cur_stream_pos = cuda_stream_allocator.impl_->stream_pos.at(cur_dev);

  auto cur_pos = cur_stream_pos.load(std::memory_order_relaxed);
  auto next_pos = (cur_pos + 1) % CUDA_STREAM_POOL_SIZE;
  while (!cur_stream_pos.compare_exchange_weak(cur_pos, next_pos,
                                               std::memory_order_acq_rel,
                                               std::memory_order_relaxed)) {
    cur_pos = cur_stream_pos.load(std::memory_order_relaxed);
    next_pos = (cur_pos + 1) % CUDA_STREAM_POOL_SIZE;
  };

  return {.stream =
              cuda_stream_allocator.impl_->stream_pool.at(cur_dev).at(cur_pos),
          .id = static_cast<DeviceId>(cur_dev)};
}

// ========================== ThreadCudaContexts ==========================
auto ThreadCudaContexts::ThreadInstance() -> ThreadCudaContexts & {
  thread_local static ThreadCudaContexts thread_cuda_contexts;
  return thread_cuda_contexts;
}

void ThreadCudaContexts::Push(CudaContext cuda_context) {
  ThreadInstance().contexts_.push(cuda_context);
}

void ThreadCudaContexts::Pop() {
  auto &thread_cuda_contexts = ThreadInstance();
  if (!thread_cuda_contexts.contexts_.empty()) {
    thread_cuda_contexts.contexts_.pop();
  }
}

auto ThreadCudaContexts::GetContext() -> CudaContext {
  auto &thread_cuda_contexts = ThreadInstance();
  if (thread_cuda_contexts.contexts_.empty()) {
    throw std::runtime_error("Emtpy cuda context.");
  }
  return thread_cuda_contexts.contexts_.top();
}

void ThreadCudaContexts::Synchronize() {
  auto &thread_cuda_contexts = ThreadInstance();
  if (!thread_cuda_contexts.contexts_.empty()) {
    TINY_LLM_CUDA_CHECK(
        cudaStreamSynchronize(ThreadInstance().contexts_.top().stream));
  }
}

void ThreadCudaContexts::SynchronizeDevice() {
  TINY_LLM_CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace tiny_llm
