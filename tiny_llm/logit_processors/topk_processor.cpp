#include "tiny_llm/logit_processors/topk_processor.hpp"
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

void top_k_kernel(float *logit, uint32_t size, uint32_t k,
                  std::vector<LogitWithId> &logit_with_id) {
  std::vector<uint32_t> id(size);
  std::iota(id.begin(), id.end(), 0);
  quick_select(logit, id.data(), 0, size - 1, k);

  for (uint32_t i = 0; i < k; ++i) {
    logit_with_id.emplace_back(logit[i], id.at(i));
  }
}
} // namespace

void TopKProcessor::apply(
    Tensor &tensor,
    std::vector<std::vector<LogitWithId>> &logit_with_id) const {
  TINY_LLM_CHECK(tensor.shape().size() == 2);
  TINY_LLM_CHECK(tensor.shape().back() >= top_k_);

  auto b = tensor.shape().at(0);
  logit_with_id.resize(b);
  for (auto &elem : logit_with_id) {
    elem.clear();
    elem.reserve(top_k_);
  }
  if (top_k_ == 0) {
    return;
  }

  auto d = tensor.shape().at(1);
  for (int64_t cur_b = 0; cur_b < b; ++cur_b) {
    auto *cur_ptr = tensor.data<float>() + (cur_b * d);
    top_k_kernel(cur_ptr, static_cast<uint32_t>(d), top_k_,
                 logit_with_id.at(cur_b));
  }
}

static_assert(LogitProcessor<TopKProcessor>);
} // namespace tiny_llm
