#include "cuda_runtime.h"
#include "tiny_llm/device_managers/cuda/cuda_context.hpp"
#include "gtest/gtest.h"

namespace tiny_llm {
TEST(DeviceManager, CudaContext) {
  ThreadCudaContexts::Push(CudaContextAllocator::CreateCudaContext());
  auto context0 = ThreadCudaContexts::GetContext();
  ThreadCudaContexts::Push(CudaContextAllocator::CreateCudaContext());
  auto context1 = ThreadCudaContexts::GetContext();

  EXPECT_TRUE(context0.stream != nullptr);
  EXPECT_TRUE(context1.stream != nullptr);
  EXPECT_TRUE(context0.id == context1.id);

  for (size_t i = 0; i < 100; ++i) {
    ThreadCudaContexts::Push(CudaContextAllocator::CreateCudaContext());
    ThreadCudaContexts::Synchronize();
    ThreadCudaContexts::Synchronize();
  }

  for (size_t i = 0; i < 200; ++i) {
    ThreadCudaContexts::Pop();
  }
}
} // namespace tiny_llm
