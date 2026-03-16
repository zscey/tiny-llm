#pragma once

#include "tiny_llm/tokenizers/tokenizers.hpp"

struct TokenizerHandle;

namespace tiny_llm {
class HuggingfaceTokenizer {
public:
  explicit HuggingfaceTokenizer(const std::string &path);
  HuggingfaceTokenizer(const HuggingfaceTokenizer &) = delete;
  auto operator=(const HuggingfaceTokenizer &) -> HuggingfaceTokenizer = delete;
  HuggingfaceTokenizer(HuggingfaceTokenizer &&) noexcept;
  auto operator=(HuggingfaceTokenizer &&) noexcept -> HuggingfaceTokenizer &;
  ~HuggingfaceTokenizer();

  auto encode(const std::string &token, bool add_special_tokens)
      -> std::vector<uint32_t>;

  auto decode(const std::vector<uint32_t> &ids, bool skip_special_tokens)
      -> std::string;

  auto get_vocab_size() -> size_t;

private:
  TokenizerHandle *handle_{};
};
} // namespace tiny_llm
