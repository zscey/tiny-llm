#include "tiny_llm/device_managers/cpu/cpu_allocator.hpp"
#include "gtest/gtest.h"
#include <cstdint>

namespace tiny_llm {
namespace {
void test_case(size_t size, size_t alignment) {
  auto buffer = CpuAllocator::Allocate(size, alignment);
  if (size > 0) {
    EXPECT_TRUE(buffer.get_ptr() != nullptr);
  }
  EXPECT_TRUE(reinterpret_cast<std::uintptr_t>(buffer.get_ptr()) % alignment ==
              0);
  EXPECT_TRUE(buffer.get_size() == size);
  EXPECT_TRUE(buffer.get_device().type == DeviceType::kCpu);
  EXPECT_TRUE(buffer.get_device().id == 0);
}
} // namespace

TEST(DeviceManager, CpuAllocator) {
  test_case(2041, 1024);
  test_case(0, 32);
}
} // namespace tiny_llm
