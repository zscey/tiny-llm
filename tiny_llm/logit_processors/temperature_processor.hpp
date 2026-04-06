#pragma once

#include "tiny_llm/logit_processors/logit_processors.hpp"
#include "tiny_llm/ops/cpu/arithmetic.hpp"

namespace tiny_llm {
class TemperatureProcessor {
public:
  explicit TemperatureProcessor(float temperature)
      : temperature_(temperature) {}

  void apply(Tensor &logit, Tensor &id, Tensor &valid_size) const {
    (void)this;
    check_params(logit, id, valid_size);
    cpu::arithmetic(logit.data<float>(), temperature_, logit.data<float>(),
                    static_cast<size_t>(logit.shape().at(0)) *
                        logit.shape().at(1),
                    ArithmeticType::kDiv);
  }

private:
  float temperature_{};
};

static_assert(LogitProcessor<TemperatureProcessor>);
} // namespace tiny_llm
