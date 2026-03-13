#pragma once
#include "cuda_runtime.h"
#include <cstdint>
#include <vector>

namespace tiny_llm {
class CudaDeviceInfos {
public:
  /**
   * @brief Get the maximum shared memory size per block on the specified
   * device.
   *
   * @param dev_id The specified device ID. If negative, it represents the
   * current device.
   * @return size_t
   */
  static auto MaxSharedMemPerBlock(int32_t dev_id = -1) -> size_t;

  static auto SharedMemPerBlockOptin(int32_t dev_id = -1) -> size_t;

private:
  friend auto get_cuda_dev_prop(int32_t dev_id) -> const cudaDeviceProp &;

  CudaDeviceInfos();
  static auto Instance() -> const CudaDeviceInfos &;

  std::vector<cudaDeviceProp> device_props_;
};
} // namespace tiny_llm
