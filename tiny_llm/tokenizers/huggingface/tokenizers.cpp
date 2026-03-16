#include "tiny_llm/tokenizers/huggingface/tokenizers.hpp"
#include "tiny_llm/common/log_and_excepts.hpp"

namespace tiny_llm {
namespace {
struct TokenizerEncodeResult {
  uint32_t *token_ids;
  size_t len;
};

struct TokenizerDecodeResult {
  char *token;
  size_t len;
};
} // namespace
} // namespace tiny_llm

extern "C" {
auto tokenizer_init(const char *path) -> TokenizerHandle *;
void tokenizer_free(TokenizerHandle *handle);
auto tokenizer_encode(TokenizerHandle *handle, const char *text,
                      bool add_special_tokens)
    -> tiny_llm::TokenizerEncodeResult;
void tokenizer_free_encode_result(tiny_llm::TokenizerEncodeResult res);
auto tokenizer_decode(TokenizerHandle *handle, const uint32_t *ids, size_t len,
                      bool skip_special_tokens)
    -> tiny_llm::TokenizerDecodeResult;
void tokenizer_free_decode_result(tiny_llm::TokenizerDecodeResult res);
auto tokenizer_vocab_size(TokenizerHandle *handle) -> size_t;
}

namespace tiny_llm {
HuggingfaceTokenizer::HuggingfaceTokenizer(const std::string &path) {
  TINY_LLM_CHECK(!path.empty());
  handle_ = tokenizer_init(path.c_str());
  TINY_LLM_CHECK(handle_);
}

HuggingfaceTokenizer::HuggingfaceTokenizer(
    HuggingfaceTokenizer &&other) noexcept {
  std::swap(handle_, other.handle_);
}

auto HuggingfaceTokenizer::operator=(HuggingfaceTokenizer &&other) noexcept
    -> HuggingfaceTokenizer & {
  if (this != std::addressof(other)) {
    std::swap(handle_, other.handle_);
  }
  return *this;
}

HuggingfaceTokenizer::~HuggingfaceTokenizer() { tokenizer_free(handle_); }

auto HuggingfaceTokenizer::encode(const std::string &token,
                                  bool add_special_tokens)
    -> std::vector<uint32_t> {
  auto tmp_res = tokenizer_encode(handle_, token.c_str(), add_special_tokens);
  std::vector<uint32_t> res(tmp_res.token_ids, tmp_res.token_ids + tmp_res.len);
  tokenizer_free_encode_result(tmp_res);
  return res;
}

auto HuggingfaceTokenizer::decode(const std::vector<uint32_t> &ids,
                                  bool skip_special_tokens) -> std::string {
  auto tmp_res =
      tokenizer_decode(handle_, ids.data(), ids.size(), skip_special_tokens);
  std::string res(tmp_res.token, tmp_res.len);
  tokenizer_free_decode_result(tmp_res);
  return res;
}

auto HuggingfaceTokenizer::get_vocab_size() -> size_t {
  return tokenizer_vocab_size(handle_);
}

static_assert(Tokenizer<HuggingfaceTokenizer>);
} // namespace tiny_llm
