#include "tiny_llm/device_managers/cuda/cuda_device_infos.hpp"
#include "gtest/gtest.h"

namespace tiny_llm {
TEST(DeviceManager, CudaDeviceInfos) {
  EXPECT_TRUE(CudaDeviceInfos::MaxSharedMemPerBlock() > 0);
}
} // namespace tiny_llm
