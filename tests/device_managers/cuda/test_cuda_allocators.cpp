#include "tiny_llm/device_managers/cuda/cuda_allocator.hpp"
#include "tiny_llm/device_managers/cuda/cuda_guards.hpp"
#include "gtest/gtest.h"
#include <cstdint>

namespace tiny_llm {
namespace {
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void test_case(size_t size) {
  auto host_buffer = CudaHostAllocator::Allocate(size);
  if (size != 0) {
    EXPECT_TRUE(host_buffer.get_ptr() != nullptr);
  }
  EXPECT_TRUE(host_buffer.get_size() == size);
  EXPECT_TRUE(host_buffer.get_device().type == DeviceType::kCudaHost);
  EXPECT_TRUE(host_buffer.get_device().id == 0);

  CudaDeviceSwitchGuard dev_guard(0);

  auto buffer = CudaAllocator::Allocate(size);
  if (size != 0) {
    EXPECT_TRUE(host_buffer.get_ptr() != nullptr);
  }
  EXPECT_TRUE(buffer.get_size() == size);
  EXPECT_TRUE(buffer.get_device().type == DeviceType::kCuda);
  EXPECT_TRUE(buffer.get_device().id == 0);
}
} // namespace

TEST(DeviceManager, CudaAllocator) {
  test_case(1024);
  test_case(0);
}

} // namespace tiny_llm
