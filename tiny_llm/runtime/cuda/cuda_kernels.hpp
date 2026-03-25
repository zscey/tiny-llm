#pragma once

#include "cuda_runtime.h"
#include "tiny_llm/runtime/common.hpp"
#include <cstddef>
#include <variant>

namespace tiny_llm::cuda {
/**
 * 1. dtype_shape_infer(...): sets the `dtype` and `cur_shape` for each output
 * desc.
 * 2. execute(...): runs the corresponding kernel in the current thread context.
 */

class SiLUKernel {
public:
  bool inplace{};

  size_t element_size{};

  /// @brief {input} -> {output}
  void dtype_shape_infer(const TensorDesc *const *input_descs,
                         TensorDesc *const *output_descs);
  void execute(const void *const *inputs, void *const *outputs) const;
};

using CudaKernel = std::variant<SiLUKernel>;
} // namespace tiny_llm::cuda
