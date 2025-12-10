#include "cuda_runtime.h"
#include "tiny_llm/device_managers/cuda/cuda_guards.hpp"
#include "gtest/gtest.h"

namespace tiny_llm {
TEST(DeviceManager, CudaGuards) {
  CudaDeviceSwitchGuard switch_guard(0);

  auto context = CudaContextAllocator::CreateCudaContext();
  {
    ThreadCudaContextsGuard context_guard(context);
    auto cur_context = ThreadCudaContexts::GetContext();

    EXPECT_TRUE(cur_context.id == context.id);
    EXPECT_TRUE(cur_context.stream == context.stream);
  }

  auto cur_context = ThreadCudaContexts::GetContext();

  EXPECT_TRUE(cur_context.stream != context.stream);
}
} // namespace tiny_llm
