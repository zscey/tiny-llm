#pragma once

#include "cuda_runtime.h"
#include "fmt/format.h"
#include "fmt/ranges.h"
#include "spdlog/spdlog.h"

#define TINY_LLM_CUDA_CHECK(exception_type, expr)                              \
  do {                                                                         \
    const cudaError_t _cur_error = (expr);                                     \
    const cudaError_t _last_error = cudaGetLastError();                        \
    if (_cur_error != cudaSuccess || _last_error != cudaSuccess) {             \
      const cudaError_t _error =                                               \
          _cur_error != cudaSuccess ? _cur_error : _last_error;                \
      auto err_msg = fmt::format("[{}]: {}.", cudaGetErrorName(_error),        \
                                 cudaGetErrorString(_error));                  \
      spdlog::error("[{}: {}]: {}", __FILE__, __LINE__, err_msg);              \
      throw exception_type(err_msg);                                           \
    }                                                                          \
  } while (false)

#define TINY_LLM_CUDA_WARN(expr)                                               \
  do {                                                                         \
    const cudaError_t _cur_error = (expr);                                     \
    if (_cur_error != cudaSuccess) {                                           \
      cudaGetLastError();                                                      \
      auto err_msg = fmt::format("[{}]: {}.", cudaGetErrorName(_cur_error),    \
                                 cudaGetErrorString(_cur_error));              \
      spdlog::warn("[{}: {}]: {}", __FILE__, __LINE__, err_msg);               \
    }                                                                          \
  } while (false)
