#pragma once

#include "tiny_llm/ops/cuda/cuda_op_common.hpp"

namespace tiny_llm::cuda {
void silu(const float *src, float *dst, size_t size, cudaStream_t stream);
}
