#pragma once

#include "tiny_llm/logit_processors/logit_processors.hpp"

namespace tiny_llm {
class TopKProcessor {
public:
  explicit TopKProcessor(uint32_t top_k) : top_k_(top_k) {}

  void apply(Tensor &logit, Tensor &id, Tensor &valid_size) const;

private:
  uint32_t top_k_{};
};

class TopPProcessor {
public:
  explicit TopPProcessor(float top_p, uint32_t min_tokens_to_keep = 1);

  void apply(Tensor &logit, Tensor &id, Tensor &valid_size) const;

private:
  float top_p_{};
  uint32_t min_tokens_to_keep_{};
};
} // namespace tiny_llm
