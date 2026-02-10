#pragma once

#include "tiny_llm/tensor/tensor.hpp"

namespace tiny_llm {
struct TensorDesc {
  DataType dtype;
  std::vector<size_t> cur_shape;
  std::vector<size_t> max_shape;
};
} // namespace tiny_llm
