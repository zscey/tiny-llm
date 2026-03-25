#include "tiny_llm/graph/param.hpp"
#include "tiny_llm/utils/visitor.hpp"

namespace tiny_llm {
namespace {
constexpr auto kInputVisitor = Visitor{
    [](const SiLUParam &) -> uint32_t { return 1; },
};

constexpr auto kInplaceVisitor = Visitor{
    [](const SiLUParam &param) -> std::vector<uint32_t> {
      if (param.inplace) {
        return {0};
      }
      return {};
    },
};
} // namespace

auto input_num(const Param &param) -> uint32_t {
  return std::visit(kInputVisitor, param);
}

auto get_inplace_input_ids(const Param &param) -> std::vector<uint32_t> {
  return std::visit(kInplaceVisitor, param);
}
} // namespace tiny_llm
