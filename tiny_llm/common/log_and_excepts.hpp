#pragma once

#include "fmt/format.h"
#include "spdlog/spdlog.h"
#include <stdexcept>

#define TINY_LLM_CHECK(expr)                                                   \
  if (!(expr)) {                                                               \
    auto err_msg = fmt::format("`{}` failed.", #expr);                         \
    spdlog::error("[{}: {}]: {}", __FILE__, __LINE__, err_msg);                \
    throw std::runtime_error(err_msg);                                         \
  }

#define TINY_LLM_CUDA_CHECK(expr)                                              \
  if ((expr) != cudaSuccess) {                                                 \
    auto status = cudaGetLastError();                                          \
    auto err_msg =                                                             \
        fmt::format("Cuda error [{}]: {}.", cudaGetErrorName(status),          \
                    cudaGetErrorString(status));                               \
    spdlog::error("[{}: {}]: {}", __FILE__, __LINE__, err_msg);                \
    throw std::runtime_error(err_msg);                                         \
  }

#define TINY_LLM_CUDA_WARN(expr)                                               \
  if ((expr) != cudaSuccess) {                                                 \
    auto status = cudaGetLastError();                                          \
    auto err_msg =                                                             \
        fmt::format("Cuda error [{}]: {}.", cudaGetErrorName(status),          \
                    cudaGetErrorString(status));                               \
    spdlog::warn("[{}: {}]: {}", __FILE__, __LINE__, err_msg);                 \
  }
