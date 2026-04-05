#pragma once

#include "tiny_llm/logit_processors/logit_processors.hpp"

namespace tiny_llm {
class TopKProcessor {
public:
  explicit TopKProcessor(uint32_t top_k) : top_k_(top_k) {}

  auto apply(Tensor &logit, Tensor &id, uint32_t valid_size) const -> uint32_t;

private:
  uint32_t top_k_{};
};
} // namespace tiny_llm
