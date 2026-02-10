#pragma once

#include "cuda_runtime.h"
#include "tiny_llm/runtime/common.hpp"
#include <cstddef>
#include <variant>

namespace tiny_llm::cuda {
struct ExecuteContext {
  cudaStream_t stream;
};

/// @brief {a, b} -> {c}
class SiLUKernel {
public:
  size_t element_size{};

  void dtype_shape_infer(const TensorDesc *const *input_descs,
                         TensorDesc *const *output_descs);
  void execute(const void *const *inputs, void *const *outputs,
               ExecuteContext &ctx);
};

using CudaKernel = std::variant<SiLUKernel>;
} // namespace tiny_llm::cuda
