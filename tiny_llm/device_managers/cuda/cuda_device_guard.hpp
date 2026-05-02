#pragma once

#include "tiny_llm/common/construct_macros.hpp"
#include <cstdint>

namespace tiny_llm {
/// @brief Set device to `target_dev` when constructing, and reset device to
/// `origin_dev_` when destroying.
class CudaDeviceSwitchGuard {
public:
  explicit CudaDeviceSwitchGuard(int32_t target_dev);

  TINY_LLM_DELETE_COPY_MOVE(CudaDeviceSwitchGuard);

  ~CudaDeviceSwitchGuard() noexcept;

private:
  int32_t origin_dev_{};
};

} // namespace tiny_llm
