#pragma once

#include "tiny_llm/tensor/tensor.hpp"
#include <string>

namespace tiny_llm {
struct TensorDesc {
  std::string name;

  DataType dtype;
  std::vector<size_t> cur_shape;
  std::vector<size_t> max_shape;
};
} // namespace tiny_llm
