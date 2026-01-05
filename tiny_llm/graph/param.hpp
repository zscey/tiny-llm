#pragma once

#include <variant>

namespace tiny_llm {
class AddParam {};

using Param = std::variant<AddParam>;
} // namespace tiny_llm
