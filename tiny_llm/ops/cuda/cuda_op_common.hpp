#pragma once

#define ThreadNum1d 512
#define ThreadNum2d 32

#define CalBlockNum(total, thread_per_block)                                   \
  (((total) + (thread_per_block) - 1) / (thread_per_block))

#include "cuda_runtime.h"
#include "tiny_llm/device_managers/cuda/cuda_context.hpp"
