#pragma once

#include "tiny_llm/logit_processors/logit_processors.hpp"

namespace tiny_llm {
class TopKProcessor {
public:
  explicit TopKProcessor(uint32_t top_k) : top_k_(top_k) {}

  void apply(Tensor &tensor,
             std::vector<std::vector<LogitWithId>> &logit_with_id) const;

private:
  uint32_t top_k_{};
};
} // namespace tiny_llm
