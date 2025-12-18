#include "tiny_llm/device_managers/cuda/cuda_allocator.hpp"
#include "tiny_llm/device_managers/cuda/cuda_guards.hpp"
#include "gtest/gtest.h"
#include <cstdint>

namespace tiny_llm {
TEST(DeviceManager, CudaAllocator) {
  auto host_buffer = CudaHostAllocator::Allocate(1024);
  EXPECT_TRUE(host_buffer.get_size() == 1024);
  EXPECT_TRUE(host_buffer.get_device().type == DeviceType::kCudaHost);
  EXPECT_TRUE(host_buffer.get_device().id == 0);

  CudaDeviceSwitchGuard dev_guard(0);

  auto buffer = CudaAllocator::Allocate(1024);
  EXPECT_TRUE(buffer.get_size() == 1024);
  EXPECT_TRUE(buffer.get_device().type == DeviceType::kCuda);
  EXPECT_TRUE(buffer.get_device().id == 0);
}

} // namespace tiny_llm
