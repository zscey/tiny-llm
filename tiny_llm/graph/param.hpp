#pragma once

#include <cstdint>
#include <variant>

namespace tiny_llm {
class SiLUParam {};

using Param = std::variant<SiLUParam>;

auto input_num(const Param &param) -> uint32_t;

auto output_num(const Param &param) -> uint32_t;
} // namespace tiny_llm
