#pragma once

#include <cstdint>

#ifdef __CUDACC__
#include "cuda_runtime.h"
#else
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#endif

namespace tiny_llm::detail {
/// @brief find the max index in the range [beg, end) which satisfy
/// `seq_separator[index] <= query_idx`
inline __host__ __device__ auto get_request_idx(uint32_t query_idx,
                                                const uint32_t *seq_separator,
                                                uint32_t beg, uint32_t end)
    -> uint32_t {
  uint32_t res{beg};

  while (beg < end) {
    auto mid = beg + ((end - beg) / 2);
    if (seq_separator[mid] <= query_idx) {
      res = mid;
      beg = mid + 1;
    } else {
      end = mid;
    }
  }

  return res;
}
} // namespace tiny_llm::detail
