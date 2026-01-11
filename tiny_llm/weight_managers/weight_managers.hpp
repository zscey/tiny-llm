#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <string>

namespace tiny_llm {
struct SliceView {
  const uint8_t *data;
  size_t len;
};

template <typename T>
concept WeightManager = requires(T t, std::string name, SliceView slice_view) {
  { t.get_tensor(name) } -> std::same_as<SliceView>;
  { t.set_tensor(name, slice_view) } -> std::same_as<void>;
};
} // namespace tiny_llm
