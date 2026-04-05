#include "tiny_llm/logit_processors/sort_processors.hpp"
#include "tiny_llm/common/log_and_excepts.hpp"
#include <numeric>

namespace tiny_llm {
namespace {
// [left, right]
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
auto partition(float *logit, uint32_t *id, uint32_t left, uint32_t right,
               uint32_t pivot_index) -> uint32_t {
  float pivot_value = logit[pivot_index];
  std::swap(logit[pivot_index], logit[right]);
  std::swap(id[pivot_index], id[right]);

  uint32_t store_index = left;
  for (uint32_t i = left; i < right; ++i) {
    if (logit[i] > pivot_value) {
      std::swap(logit[store_index], logit[i]);
      std::swap(id[store_index], id[i]);
      ++store_index;
    }
  }

  std::swap(logit[store_index], logit[right]);
  std::swap(id[store_index], id[right]);

  return store_index;
}

// [left, right]
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters, misc-no-recursion)
void quick_select(float *logit, uint32_t *id, uint32_t left, uint32_t right,
                  uint32_t k) {
  if (left >= right) {
    return;
  }

  auto mid = left + ((right - left) / 2);
  auto pivot_index = partition(logit, id, left, right, mid);

  if (pivot_index == k || pivot_index + 1 == k) {
    return;
  }

  if (pivot_index > k) {
    quick_select(logit, id, left, pivot_index - 1, k);
  } else {
    quick_select(logit, id, pivot_index + 1, right, k);
  }
}
} // namespace

auto TopKProcessor::apply(Tensor &logit, Tensor &id, uint32_t valid_size) const
    -> uint32_t {
  TINY_LLM_CHECK(logit.dtype() == DataType::kFloat32);
  TINY_LLM_CHECK(id.dtype() == DataType::kUint32);
  TINY_LLM_CHECK(logit.shape().size() == 2);
  TINY_LLM_CHECK(logit.shape() == id.shape());
  TINY_LLM_CHECK(logit.shape().back() >= valid_size);
  TINY_LLM_CHECK(valid_size >= top_k_);

  if (top_k_ == 0) {
    return 0;
  }

  for (int64_t b = 0, b_end = logit.shape().at(0); b < b_end; ++b) {
    auto shift = b * logit.shape().at(1);
    quick_select(logit.data<float>() + shift, id.data<uint32_t>() + shift, 0,
                 valid_size - 1, top_k_);
  }

  return top_k_;
}

static_assert(LogitProcessor<TopKProcessor>);
} // namespace tiny_llm
