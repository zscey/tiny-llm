#pragma once

#include "tiny_llm/common/common_macros.hpp"
#include "tiny_llm/tensor/tensor.hpp"
#include <memory>

namespace tiny_llm {
struct LogitWithId {
  float logit{};
  uint32_t idx{};
};

template <typename T>
concept LogitProcessor =
    std::move_constructible<T> &&
    requires(T t, Tensor &tensor,
             std::vector<std::vector<LogitWithId>> &logit_with_id) {
      { t.apply(tensor, logit_with_id) } -> std::same_as<void>;
    };

class LogitProcessorWrapper {
  struct Concept {
    Concept() = default;
    TINY_LLM_DEFAULT_COPY_MOVE(Concept);
    virtual ~Concept() = default;

    virtual void
    apply(Tensor &tensor,
          std::vector<std::vector<LogitWithId>> &logit_with_id) = 0;
  };

  template <LogitProcessor T> struct Container final : public Concept {
    T logit_processor;

    explicit Container(T &&t) : logit_processor(std::move(t)) {}
    TINY_LLM_DEFAULT_COPY_MOVE(Container)
    ~Container() override = default;

    void apply(Tensor &tensor,
               std::vector<std::vector<LogitWithId>> &logit_with_id) override {
      logit_processor.apply(tensor, logit_with_id);
    }
  };

  std::unique_ptr<Concept> wrapper_;

public:
  template <LogitProcessor T>
  explicit LogitProcessorWrapper(T &&t)
      : wrapper_(std::make_unique<Container<T>>(std::forward<T>(t))) {}

  void apply(Tensor &tensor,
             std::vector<std::vector<LogitWithId>> &logit_with_id) {
    wrapper_->apply(tensor, logit_with_id);
  }
};
} // namespace tiny_llm
