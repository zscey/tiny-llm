#include "cuda_runtime.h"
#include "tiny_llm/device_managers/cuda/cuda_context.hpp"
#include "gtest/gtest.h"
#include <thread>
namespace tiny_llm {
TEST(DeviceManager, CudaContext) {
  ThreadCudaContexts::Push(CudaContextAllocator::CreateCudaContext());
  auto context0 = ThreadCudaContexts::GetContext();
  ThreadCudaContexts::Push(CudaContextAllocator::CreateCudaContext());
  auto context1 = ThreadCudaContexts::GetContext();

  EXPECT_TRUE(context0.stream != nullptr);
  EXPECT_TRUE(context1.stream != nullptr);
  EXPECT_TRUE(context0.id == context1.id);
  EXPECT_TRUE(context0.stream != context1.stream);

  for (size_t i = 0; i < 100; ++i) {
    ThreadCudaContexts::Push(CudaContextAllocator::CreateCudaContext());
    ThreadCudaContexts::Synchronize();
    ThreadCudaContexts::SynchronizeDevice();
  }

  for (size_t i = 0; i < 200; ++i) {
    ThreadCudaContexts::Pop();
  }

  std::vector<std::jthread> threads(8);
  for (size_t i = 0; i < 8; ++i) {
    threads.at(i) = std::jthread([]() -> void {
      for (size_t i = 0; i < 100; ++i) {
        CudaContextAllocator::CreateCudaContext();
      };
    });
  }
}
} // namespace tiny_llm
