#pragma once

#include "tiny_llm/common/common_macros.hpp"
#include "tiny_llm/tensor/tensor.hpp"
#include <memory>

namespace tiny_llm {
template <typename T>
concept LogitProcessor =
    std::move_constructible<T> &&
    requires(T t, Tensor &logit, Tensor &id, uint32_t valid_size) {
      { t.apply(logit, id, valid_size) } -> std::same_as<uint32_t>;
    };

class LogitProcessorWrapper {
  struct Concept {
    Concept() = default;
    TINY_LLM_DEFAULT_COPY_MOVE(Concept);
    virtual ~Concept() = default;

    virtual auto apply(Tensor &logit, Tensor &id, uint32_t valid_size)
        -> uint32_t = 0;
  };

  template <LogitProcessor T> struct Container final : public Concept {
    T logit_processor;

    explicit Container(T &&t) : logit_processor(std::move(t)) {}
    TINY_LLM_DEFAULT_COPY_MOVE(Container)
    ~Container() override = default;

    auto apply(Tensor &logit, Tensor &id, uint32_t valid_size)
        -> uint32_t override {
      return logit_processor.apply(logit, id, valid_size);
    }
  };

  std::unique_ptr<Concept> wrapper_;

public:
  template <LogitProcessor T>
  explicit LogitProcessorWrapper(T &&t)
      : wrapper_(std::make_unique<Container<T>>(std::forward<T>(t))) {}

  auto apply(Tensor &logit, Tensor &id, uint32_t valid_size) -> uint32_t {
    return wrapper_->apply(logit, id, valid_size);
  }
};
} // namespace tiny_llm
