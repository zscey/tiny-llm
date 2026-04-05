#pragma once

#include "tiny_llm/logit_processors/logit_processors.hpp"

namespace tiny_llm {
class ArgmaxProcessor {
public:
  void apply(Tensor &logit, Tensor &id, Tensor &valid_size) const;
};
} // namespace tiny_llm
