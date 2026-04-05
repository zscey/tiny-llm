#pragma once

#include "tiny_llm/logit_processors/logit_processors.hpp"
#include <random>

namespace tiny_llm {
class MultinomialProcessor {
public:
  MultinomialProcessor();

  void apply(Tensor &logit, Tensor &id, Tensor &valid_size);

private:
  std::mt19937 seed_;
  std::uniform_real_distribution<> dis_{0.F, 1.F};
};
} // namespace tiny_llm
