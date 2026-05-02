#pragma once

#include "fmt/format.h"
#include "fmt/ranges.h"
#include "spdlog/spdlog.h"

#define TINY_LLM_CHECK(exception_type, expr)                                   \
  do {                                                                         \
    if (!(expr)) {                                                             \
      auto err_msg = fmt::format("`{}` failed.", #expr);                       \
      spdlog::error("[{}: {}]: {}", __FILE__, __LINE__, err_msg);              \
      throw exception_type(err_msg);                                           \
    }                                                                          \
  } while (false)

#define TINY_LLM_THROW_ERROR(exception_type, ...)                              \
  do {                                                                         \
    auto err_msg = fmt::format(__VA_ARGS__);                                   \
    spdlog::error("[{}: {}]: {}", __FILE__, __LINE__, err_msg);                \
    throw exception_type(err_msg);                                             \
  } while (false)
