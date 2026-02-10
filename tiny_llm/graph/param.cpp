#include "tiny_llm/graph/param.hpp"
#include "tiny_llm/utils/visitor.hpp"

namespace tiny_llm {
namespace {
constexpr auto kInputVisitor =
    Visitor{[](const SiLUParam &) -> uint32_t { return 1; }};

constexpr auto kOutputVisitor =
    Visitor{[](const SiLUParam &) -> uint32_t { return 1; }};
} // namespace

auto input_num(const Param &param) -> uint32_t {
  return std::visit(kInputVisitor, param);
}

auto output_num(const Param &param) -> uint32_t {
  return std::visit(kOutputVisitor, param);
}
} // namespace tiny_llm
