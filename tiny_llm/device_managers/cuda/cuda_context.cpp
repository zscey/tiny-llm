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

CudaContextAllocator::~CudaContextAllocator() noexcept = default;
// CudaContextAllocator::~CudaContextAllocator() noexcept {
// int32_t dev_num{};
// TINY_LLM_CUDA_WARN(cudaGetDeviceCount(&dev_num));

// int32_t cur_dev{};
// TINY_LLM_CUDA_WARN(cudaGetDevice(&cur_dev));

// for (int32_t dev_id = 0, dev_id_end = std::min(dev_num,
// MAX_CUDA_DEVICE_NUM);
//      dev_id < dev_id_end; ++dev_id) {
//   TINY_LLM_CUDA_WARN(cudaSetDevice(dev_id));

//   auto &streams = impl_->stream_pool.at(dev_id);
//   for (size_t stream_id = 0; stream_id < CUDA_STREAM_POOL_SIZE;
//   ++stream_id) {
//     TINY_LLM_CUDA_WARN(cudaStreamDestroy(streams.at(stream_id)));
//   }
// }

// TINY_LLM_CUDA_WARN(cudaSetDevice(cur_dev));
// }

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
  int32_t dev_id{};
  TINY_LLM_CUDA_CHECK(cudaGetDevice(&dev_id));
  TINY_LLM_CHECK(dev_id == cuda_context.id);

  ThreadInstance().impl_->contexts.at(cuda_context.id).push(cuda_context);
}

void ThreadCudaContexts::Pop() {
  auto &thread_cuda_contexts = ThreadInstance();

  int32_t dev_id{};
  TINY_LLM_CUDA_CHECK(cudaGetDevice(&dev_id));
  auto &cur_contexts = thread_cuda_contexts.impl_->contexts.at(dev_id);
  if (!cur_contexts.empty()) {
    cur_contexts.pop();
  }
}

auto ThreadCudaContexts::GetContext() -> CudaContext {
  auto &thread_cuda_contexts = ThreadInstance();

  int32_t dev_id{};
  TINY_LLM_CUDA_CHECK(cudaGetDevice(&dev_id));
  auto &cur_contexts = thread_cuda_contexts.impl_->contexts.at(dev_id);
  if (cur_contexts.empty()) {
    cur_contexts.push(CudaContextAllocator::CreateCudaContext());
  }

  return cur_contexts.top();
}

void ThreadCudaContexts::Synchronize() {
  auto &thread_cuda_contexts = ThreadInstance();

  int32_t dev_id{};
  TINY_LLM_CUDA_CHECK(cudaGetDevice(&dev_id));
  auto &cur_contexts = thread_cuda_contexts.impl_->contexts.at(dev_id);
  if (!cur_contexts.empty()) {
    TINY_LLM_CUDA_CHECK(cudaStreamSynchronize(cur_contexts.top().stream));
  }
}

void ThreadCudaContexts::SynchronizeDevice() {
  TINY_LLM_CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace tiny_llm
