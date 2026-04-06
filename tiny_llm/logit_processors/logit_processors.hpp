#pragma once

#include "tiny_llm/common/common_macros.hpp"
#include "tiny_llm/common/log_and_excepts.hpp"
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

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
inline void check_params(const Tensor &logit, const Tensor &id,
                         const Tensor &valid_size) {
  TINY_LLM_CHECK(logit.dtype() == DataType::kFloat32);
  TINY_LLM_CHECK(id.dtype() == DataType::kUint32);
  TINY_LLM_CHECK(valid_size.dtype() == DataType::kUint32);
  TINY_LLM_CHECK(logit.shape().size() == 2);
  TINY_LLM_CHECK(logit.shape() == id.shape());
  TINY_LLM_CHECK(valid_size.shape() ==
                 std::vector<int64_t>{logit.shape().at(0)});
}
} // namespace tiny_llm
