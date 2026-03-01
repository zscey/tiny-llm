#pragma once

#define CalBlockNum(total, thread_per_block)                                   \
  (((total) + (thread_per_block) - 1) / (thread_per_block))

#include "cuda_runtime.h"
#include "tiny_llm/device_managers/cuda/cuda_context.hpp"
