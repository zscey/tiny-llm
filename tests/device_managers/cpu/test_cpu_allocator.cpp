#include "tiny_llm/device_managers/cpu/cpu_allocator.hpp"
#include "gtest/gtest.h"
#include <cstdint>

namespace tiny_llm {
TEST(DeviceManager, CpuAllocator) {
  auto buffer = CpuAllocator::Allocate(2048, 1024);
  EXPECT_TRUE(reinterpret_cast<std::uintptr_t>(buffer.get_ptr()) % 1024 == 0);
  EXPECT_TRUE(buffer.get_size() == 2048);
  EXPECT_TRUE(buffer.get_device().type == DeviceType::kCpu);
  EXPECT_TRUE(buffer.get_device().id == 0);
}
} // namespace tiny_llm
