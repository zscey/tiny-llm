#pragma once

#include <cstdint>

namespace tiny_llm {
enum class ArithmeticType : std::uint8_t {
  kAdd,
  kMul,
  kSub,
  kDiv,
};
}
