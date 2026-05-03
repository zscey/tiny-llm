#include "tiny_llm/graph/param.hpp"
#include "tiny_llm/utils/visitor.hpp"

namespace tiny_llm {
namespace {
constexpr auto kInputVisitor = Visitor{
    [](const SiLUParam &) -> uint32_t { return extern_input_num<SiLUParam>(); },
    [](const EmbeddingParam &) -> uint32_t {
      return extern_input_num<EmbeddingParam>() + 1;
    },
    [](const RopeParam &) -> uint32_t { return extern_input_num<RopeParam>(); },
    [](const RMSNormParam &) -> uint32_t {
      return extern_input_num<RMSNormParam>() + 1;
    },
    [](const AddParam &) -> uint32_t { return extern_input_num<AddParam>(); },
    [](const MulParam &) -> uint32_t { return extern_input_num<MulParam>(); },
    [](const LinearParam &param) -> uint32_t {
      return extern_input_num<LinearParam>() + 1 + (param.bias ? 1 : 0);
    },
    [](const CausalAttentionParam &param) -> uint32_t {
      return extern_input_num<CausalAttentionParam>() + 4 +
             (param.bias ? 4 : 0);
    },
    [](const SliceLinearParam &param) -> uint32_t {
      return extern_input_num<SliceLinearParam>() + 1 + (param.bias ? 1 : 0);
    },
};

constexpr auto kInplaceVisitor = Visitor{
    [](const SiLUParam &param) -> std::vector<uint32_t> {
      if (param.inplace) {
        return {0};
      }
      return {};
    },
    [](const EmbeddingParam &) -> std::vector<uint32_t> { return {}; },
    [](const RopeParam &) -> std::vector<uint32_t> { return {}; },
    [](const RMSNormParam &param) -> std::vector<uint32_t> {
      if (param.inplace) {
        return {0};
      }
      return {};
    },
    [](const AddParam &param) -> std::vector<uint32_t> {
      if (param.out_idx == param.left_idx) {
        return {param.left_idx};
      }
      if (param.out_idx == param.right_idx) {
        return {param.right_idx};
      }
      return {};
    },
    [](const MulParam &param) -> std::vector<uint32_t> {
      if (param.out_idx == param.left_idx) {
        return {param.left_idx};
      }
      if (param.out_idx == param.right_idx) {
        return {param.right_idx};
      }
      return {};
    },
    [](const LinearParam &) -> std::vector<uint32_t> { return {}; },
    [](const CausalAttentionParam &) -> std::vector<uint32_t> { return {}; },
    [](const SliceLinearParam &) -> std::vector<uint32_t> { return {}; },
};
} // namespace

auto input_num(const Param &param) -> uint32_t {
  return std::visit(kInputVisitor, param);
}

auto output_num(const Param &param) -> uint32_t {
  return std::visit(
      [](const auto &p) -> uint32_t { return output_num<decltype(p)>(); },
      param);
}

auto get_inplace_input_ids(const Param &param) -> std::vector<uint32_t> {
  return std::visit(kInplaceVisitor, param);
}
} // namespace tiny_llm
