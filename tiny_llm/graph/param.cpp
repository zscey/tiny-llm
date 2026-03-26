#include "tiny_llm/graph/param.hpp"
#include "tiny_llm/utils/visitor.hpp"

namespace tiny_llm {
namespace {
constexpr auto kInputVisitor = Visitor{
    [](const SiLUParam &) -> uint32_t { return 1; },
    [](const EmbeddingParam &) -> uint32_t { return 2; },
    // [](const RopeParam &) -> uint32_t { return 0; },
    // [](const RMSNormParam &) -> uint32_t { return 2; },
    // [](const LinearParam &param) -> uint32_t { return param.bias ? 3 : 2; },
    // [](const AttentionParam &param) -> uint32_t { return param.bias ? 12 : 8;
    // },
    // [](const AddParam &) -> uint32_t { return 2; },
    // [](const MulParam &) -> uint32_t { return 2; },
};

constexpr auto kInplaceVisitor = Visitor{
    [](const SiLUParam &param) -> std::vector<uint32_t> {
      if (param.inplace) {
        return {0};
      }
      return {};
    },
    [](const EmbeddingParam &) -> std::vector<uint32_t> { return {}; },
    // [](const RopeParam &) -> std::vector<uint32_t> { return {}; },
    // [](const RMSNormParam &param) -> std::vector<uint32_t> {
    //   if (param.inplace) {
    //     return {0};
    //   }
    //   return {};
    // },
    // [](const LinearParam &) -> std::vector<uint32_t> { return {}; },
    // [](const AttentionParam &) -> std::vector<uint32_t> { return {}; },
    // [](const AddParam &param) -> std::vector<uint32_t> {
    //   if (param.out_idx == param.left_idx) {
    //     return {param.left_idx};
    //   }
    //   if (param.out_idx == param.right_idx) {
    //     return {param.right_idx};
    //   }
    //   return {};
    // },
    // [](const MulParam &param) -> std::vector<uint32_t> {
    //   if (param.out_idx == param.left_idx) {
    //     return {param.left_idx};
    //   }
    //   if (param.out_idx == param.right_idx) {
    //     return {param.right_idx};
    //   }
    //   return {};
    // },
};
} // namespace

auto input_num(const Param &param) -> uint32_t {
  return std::visit(kInputVisitor, param);
}

auto get_inplace_input_ids(const Param &param) -> std::vector<uint32_t> {
  return std::visit(kInplaceVisitor, param);
}
} // namespace tiny_llm
