#include "tiny_llm/device_managers/cpu_allocator.hpp"
#include "gtest/gtest.h"
#include <cstdint>

TEST(DeviceManager, CpuAllocator) {
  auto buffer = tiny_llm::CpuAllocator::Allocate(321, 1024);
  EXPECT_TRUE(reinterpret_cast<std::uintptr_t>(buffer.get_ptr()) % 1024 == 0);
}
