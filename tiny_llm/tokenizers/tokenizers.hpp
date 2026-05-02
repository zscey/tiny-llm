#pragma once

#include "tiny_llm/common/construct_macros.hpp"
#include <concepts>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tiny_llm {
template <typename T>
concept Tokenizer =
    std::move_constructible<T> &&
    requires(T t, std::string token, bool add_special_tokens,
             std::vector<uint32_t> ids, bool skip_special_tokens) {
      { t.get_vocab_size() } -> std::same_as<size_t>;
      {
        t.encode(token, add_special_tokens)
      } -> std::same_as<std::vector<uint32_t>>;
      { t.decode(ids, skip_special_tokens) } -> std::same_as<std::string>;
    };

class TokenizerWrapper {
  struct Concept {
    Concept() = default;
    TINY_LLM_DEFAULT_COPY_MOVE(Concept);
    virtual ~Concept() = default;

    virtual auto encode(const std::string &token, bool add_special_tokens)
        -> std::vector<uint32_t> = 0;
    virtual auto decode(const std::vector<uint32_t> &ids,
                        bool skip_special_tokens) -> std::string = 0;
    virtual auto get_vocab_size() -> size_t = 0;
  };

  template <Tokenizer T> struct Container final : public Concept {
    T tokenizer;

    explicit Container(T &&t) : tokenizer(std::move(t)) {}
    TINY_LLM_DEFAULT_COPY_MOVE(Container);
    ~Container() override = default;

    auto encode(const std::string &token, bool add_special_tokens)
        -> std::vector<uint32_t> override {
      return tokenizer.encode(token, add_special_tokens);
    }

    auto decode(const std::vector<uint32_t> &ids, bool skip_special_tokens)
        -> std::string override {
      return tokenizer.decode(ids, skip_special_tokens);
    }

    auto get_vocab_size() -> size_t override {
      return tokenizer.get_vocab_size();
    }
  };

  std::unique_ptr<Concept> wrapper_;

public:
  template <Tokenizer T>
    requires(!std::is_same_v<std::decay_t<T>, TokenizerWrapper>)
  explicit TokenizerWrapper(T &&t)
      : wrapper_(
            std::make_unique<Container<std::decay_t<T>>>(std::forward<T>(t))) {}

  auto encode(const std::string &token, bool add_special_tokens)
      -> std::vector<uint32_t> {
    return wrapper_->encode(token, add_special_tokens);
  }

  auto decode(const std::vector<uint32_t> &ids, bool skip_special_tokens)
      -> std::string {
    return wrapper_->decode(ids, skip_special_tokens);
  }

  auto get_vocab_size() -> size_t { return wrapper_->get_vocab_size(); }
};
} // namespace tiny_llm
