#pragma once

#include "tiny_llm/common/common_macros.hpp"
#include "tiny_llm/tensor/tensor.hpp"
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace tiny_llm {
/**
 * @brief SliceView represents a non-owning window into a tensor's memory.
 * Lifetime is tied to the WeightManager providing the data.
 */
struct SliceView {
  DataType dtype;
  std::vector<int64_t> shape;
  const void *data;
  size_t data_len; // byte size
};

template <typename T>
concept WeightManager =
    std::move_constructible<T> &&
    requires(const T ct, T t, std::string name, SliceView slice_view) {
      { ct.get_tensor(name) } -> std::same_as<SliceView>;
      { t.set_tensor(name, slice_view) } -> std::same_as<void>;
    };

class WeightManagerWrapper {
  struct Concept {
    Concept() = default;
    TINY_LLM_DEFAULT_COPY_MOVE(Concept);
    virtual ~Concept() = default;

    [[nodiscard]] virtual auto get_tensor(const std::string &name) const
        -> SliceView = 0;
    virtual void set_tensor(std::string name, SliceView slice_view) = 0;
  };

  template <WeightManager T> struct Container final : public Concept {
    T weight_manager;

    explicit Container(T &&t) : weight_manager(std::move(t)) {}
    TINY_LLM_DEFAULT_COPY_MOVE(Container);
    ~Container() override = default;

    [[nodiscard]] auto get_tensor(const std::string &name) const
        -> SliceView override {
      return weight_manager.get_tensor(name);
    }
    void set_tensor(std::string name, SliceView slice_view) override {
      weight_manager.set_tensor(std::move(name), std::move(slice_view));
    }
  };

  std::unique_ptr<Concept> wrapper_;

public:
  template <WeightManager T>
    requires(!std::is_same_v<std::decay_t<T>, WeightManagerWrapper>)
  explicit WeightManagerWrapper(T &&t)
      : wrapper_(
            std::make_unique<Container<std::decay_t<T>>>(std::forward<T>(t))) {}

  [[nodiscard]] auto get_tensor(const std::string &name) const -> SliceView {
    return wrapper_->get_tensor(name);
  }

  void set_tensor(std::string name, SliceView slice_view) {
    wrapper_->set_tensor(std::move(name), std::move(slice_view));
  }
};
} // namespace tiny_llm
