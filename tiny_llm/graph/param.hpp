#pragma once

#include <cstdint>
#include <variant>
#include <vector>

namespace tiny_llm {
class SiLUParam {
public:
  bool inplace{false};
};

class EmbeddingParam {
public:
  uint32_t num_embeddings{};
  uint32_t hidden_size{};
};

// class RopeParam {
// public:
//   uint32_t max_len{};
//   uint32_t head_dim{};
//   double theta{};
// };

// class RMSNormParam {
// public:
//   uint32_t hidden_size{};
//   float eps{};
//   bool inplace{false};
// };

// class LinearParam {
// public:
//   uint32_t in_dim{};
//   uint32_t out_dim{};
//   bool bias{false};
// };

// class AttentionParam {
// public:
//   uint32_t head_dim{};
//   uint32_t q_head{};
//   uint32_t kv_head{};
//   bool bias{false};
// };

// class AddParam {
// public:
//   uint32_t left_idx{0};
//   uint32_t right_idx{1};
//   uint32_t out_idx{2};
// };

// class MulParam {
// public:
//   uint32_t left_idx{0};
//   uint32_t right_idx{1};
//   uint32_t out_idx{2};
// };

using Param = std::variant<SiLUParam, EmbeddingParam>;

namespace details {
template <typename T> constexpr auto extern_input_num_impl() -> uint32_t {
  return 1;
}
// template <> constexpr auto extern_input_num_impl<RopeParam>() -> uint32_t {
//   return 0;
// }
// template <> constexpr auto extern_input_num_impl<AttentionParam>() ->
// uint32_t {
//   return 4;
// }
// template <> constexpr auto extern_input_num_impl<AddParam>() -> uint32_t {
//   return 2;
// }
// template <> constexpr auto extern_input_num_impl<MulParam>() -> uint32_t {
//   return 2;
// }

template <typename T> constexpr auto output_num_impl() -> uint32_t { return 1; }
// template <> constexpr auto output_num_impl<RopeParam>() -> uint32_t {
//   return 2;
// }
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
