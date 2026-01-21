#pragma once

#define ThreadNum1d 512
#define ThreadNum2dX 32
#define ThreadNum2dY 16

#define CalBlockNum(total, thread_per_block)                                   \
  (((total) + (thread_per_block) - 1) / (thread_per_block))

#include "cuda_runtime.h"
