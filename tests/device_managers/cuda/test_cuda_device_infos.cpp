#include "tiny_llm/device_managers/cuda/cuda_device_infos.hpp"
#include "gtest/gtest.h"

namespace tiny_llm {
TEST(DeviceManager, CudaDeviceInfos) {
  EXPECT_TRUE(CudaDeviceInfos::MaxSharedMemPerBlock() > 0);
  EXPECT_TRUE(CudaDeviceInfos::SharedMemPerBlockOptin() >=
              CudaDeviceInfos::MaxSharedMemPerBlock());
}
} // namespace tiny_llm
