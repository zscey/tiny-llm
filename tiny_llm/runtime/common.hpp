#pragma once

#include "tiny_llm/tensor/tensor.hpp"

namespace tiny_llm {
template <class... Ts> struct Visitor : Ts... {
  using Ts::operator()...;
};
template <class... Ts> Visitor(Ts...) -> Visitor<Ts...>;

struct TensorDesc {
  DataType dtype;
  std::vector<size_t> cur_shape;
  std::vector<size_t> max_shape;
};
} // namespace tiny_llm
