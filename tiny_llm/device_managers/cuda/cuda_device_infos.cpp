#include "tiny_llm/device_managers/cuda/cuda_device_infos.hpp"
#include "tiny_llm/common/log_and_excepts.hpp"

namespace tiny_llm {
CudaDeviceInfos::CudaDeviceInfos() {
  int32_t dev_num{};
  TINY_LLM_CUDA_CHECK(cudaGetDeviceCount(&dev_num));
  device_props_.resize(dev_num);

  for (int32_t i = 0; i < dev_num; ++i) {
    TINY_LLM_CUDA_CHECK(cudaGetDeviceProperties(&device_props_.at(i), i));
  }
}

auto CudaDeviceInfos::Instance() -> const CudaDeviceInfos & {
  static CudaDeviceInfos device_infos;
  return device_infos;
}

auto get_cuda_dev_prop(int32_t dev_id) -> const cudaDeviceProp & {
  const auto &device_infos = CudaDeviceInfos::Instance();

  if (dev_id < 0) {
    TINY_LLM_CUDA_CHECK(cudaGetDevice(&dev_id));
  }
  TINY_LLM_CHECK(static_cast<size_t>(dev_id) <
                 device_infos.device_props_.size());
  return device_infos.device_props_.at(dev_id);
}

auto CudaDeviceInfos::MaxSharedMemPerBlock(int32_t dev_id) -> size_t {
  return get_cuda_dev_prop(dev_id).sharedMemPerBlock;
}

auto CudaDeviceInfos::SharedMemPerBlockOptin(int32_t dev_id) -> size_t {
  return get_cuda_dev_prop(dev_id).sharedMemPerBlockOptin;
}
} // namespace tiny_llm
