#pragma once

#include "tiny_llm/common/common_macros.hpp"
#include "tiny_llm/tensor/tensor.hpp"
#include <memory>

namespace tiny_llm {
template <typename T>
concept LogitProcessor =
    std::move_constructible<T> &&
    requires(T t, Tensor &logit, Tensor &id, Tensor &valid_size) {
      { t.apply(logit, id, valid_size) } -> std::same_as<void>;
    };

class LogitProcessorWrapper {
  struct Concept {
    Concept() = default;
    TINY_LLM_DEFAULT_COPY_MOVE(Concept);
    virtual ~Concept() = default;

    virtual void apply(Tensor &logit, Tensor &id, Tensor &valid_size) = 0;
  };

  template <LogitProcessor T> struct Container final : public Concept {
    T logit_processor;

    explicit Container(T &&t) : logit_processor(std::move(t)) {}
    TINY_LLM_DEFAULT_COPY_MOVE(Container)
    ~Container() override = default;

    void apply(Tensor &logit, Tensor &id, Tensor &valid_size) override {
      logit_processor.apply(logit, id, valid_size);
    }
  };

  std::unique_ptr<Concept> wrapper_;

public:
  template <LogitProcessor T>
  explicit LogitProcessorWrapper(T &&t)
      : wrapper_(std::make_unique<Container<T>>(std::forward<T>(t))) {}

  void apply(Tensor &logit, Tensor &id, Tensor &valid_size) {
    wrapper_->apply(logit, id, valid_size);
  }
};
} // namespace tiny_llm
