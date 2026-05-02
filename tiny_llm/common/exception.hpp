#pragma once

#include "tiny_llm/common/construct_macros.hpp"
#include <exception>
#include <string>

namespace tiny_llm {
class TinyLlmException : public std::exception {
public:
  explicit TinyLlmException(std::string msg) : msg_(std::move(msg)) {}
  TINY_LLM_DEFAULT_COPY_MOVE(TinyLlmException);
  ~TinyLlmException() override = default;

  [[nodiscard]] auto what() const noexcept -> const char * override {
    return msg_.c_str();
  }

private:
  std::string msg_;
};

// NOLINTBEGIN(cppcoreguidelines-macro-usage,bugprone-macro-parentheses)
#define DEFINE_TINY_LLM_EXCEPTION(DerivedName)                                 \
  class DerivedName : public TinyLlmException {                                \
    using TinyLlmException::TinyLlmException;                                  \
  }

DEFINE_TINY_LLM_EXCEPTION(RuntimeError);
DEFINE_TINY_LLM_EXCEPTION(InvalidArgumentError);
DEFINE_TINY_LLM_EXCEPTION(NotImplementedError);
DEFINE_TINY_LLM_EXCEPTION(CudaError);

#undef DEFINE_TINY_LLM_EXCEPTION
// NOLINTEND(cppcoreguidelines-macro-usage,bugprone-macro-parentheses)
} // namespace tiny_llm
