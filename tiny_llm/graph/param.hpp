#pragma once

#include <cstdint>
#include <variant>
#include <vector>

namespace tiny_llm {
class SiLUParam {
public:
  bool inplace{false};
};

using Param = std::variant<SiLUParam>;

namespace details {
template <typename T> constexpr auto extern_input_num_impl() -> uint32_t {
  return 1;
}

template <typename T> constexpr auto output_num_impl() -> uint32_t { return 1; }
} // namespace details

template <typename T> constexpr auto extern_input_num() -> uint32_t {
  return details::extern_input_num_impl<std::remove_cvref_t<T>>();
}
auto input_num(const Param &param) -> uint32_t;

template <typename T> constexpr auto output_num() -> uint32_t {
  return details::output_num_impl<std::remove_cvref_t<T>>();
}
inline auto output_num(const Param &param) -> uint32_t {
  return std::visit(
      [](const auto &p) -> uint32_t { return output_num<decltype(p)>(); },
      param);
}

auto get_inplace_input_ids(const Param &param) -> std::vector<uint32_t>;
} // namespace tiny_llm
