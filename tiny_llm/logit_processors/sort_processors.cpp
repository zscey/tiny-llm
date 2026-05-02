#include "tiny_llm/logit_processors/sort_processors.hpp"
#include "tiny_llm/common/checks.hpp"
#include "tiny_llm/common/exception.hpp"
#include <numeric>

namespace tiny_llm {
namespace {
// [left, right]
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
auto partition(float *logit, uint32_t *id, int32_t left, int32_t right,
               int32_t pivot_index) -> int32_t {
  float pivot_value = logit[pivot_index];
  std::swap(logit[pivot_index], logit[right]);
  std::swap(id[pivot_index], id[right]);

  int32_t store_index = left;
  for (int32_t i = left; i < right; ++i) {
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
void quick_select(float *logit, uint32_t *id, int32_t left, int32_t right,
                  int32_t k) {
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

void TopKProcessor::apply(Tensor &logit, Tensor &id, Tensor &valid_size) const {
  check_params(logit, id, valid_size);

  int64_t batch = logit.shape().at(0);
  int64_t dim = logit.shape().at(1);
  for (int64_t i = 0; i < batch; ++i) {
    TINY_LLM_CHECK(tiny_llm::InvalidArgumentError,
                   valid_size.data<uint32_t>()[i] <= dim);
    TINY_LLM_CHECK(tiny_llm::InvalidArgumentError,
                   valid_size.data<uint32_t>()[i] >= top_k_);
  }

  if (top_k_ == 0) {
    for (int64_t i = 0; i < batch; ++i) {
      valid_size.data<uint32_t>()[i] = top_k_;
    }
    return;
  }

  for (int64_t i = 0; i < batch; ++i) {
    auto shift = i * dim;
    quick_select(logit.data<float>() + shift, id.data<uint32_t>() + shift, 0,
                 static_cast<int32_t>(valid_size.data<uint32_t>()[i]) - 1,
                 static_cast<int32_t>(top_k_));
    valid_size.data<uint32_t>()[i] = top_k_;
  }
}

static_assert(LogitProcessor<TopKProcessor>);

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
TopPProcessor::TopPProcessor(float top_p, uint32_t min_tokens_to_keep) {
  TINY_LLM_CHECK(tiny_llm::InvalidArgumentError, top_p >= 0.F);
  TINY_LLM_CHECK(tiny_llm::InvalidArgumentError, top_p <= 1.F);
  TINY_LLM_CHECK(tiny_llm::InvalidArgumentError, min_tokens_to_keep > 0);

  top_p_ = top_p;
  min_tokens_to_keep_ = min_tokens_to_keep;
}

namespace {
// [left, right]
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters,misc-no-recursion)
void quick_sort(float *logit, uint32_t *id, int32_t left, int32_t right) {
  if (left >= right) {
    return;
  }

  auto mid = left + ((right - left) / 2);
  auto pivot_index = partition(logit, id, left, right, mid);
  quick_sort(logit, id, left, pivot_index - 1);
  quick_sort(logit, id, pivot_index + 1, right);
}

} // namespace

void TopPProcessor::apply(Tensor &logit, Tensor &id, Tensor &valid_size) const {
  check_params(logit, id, valid_size);

  int64_t batch = logit.shape().at(0);
  int64_t dim = logit.shape().at(1);
  for (int64_t i = 0; i < batch; ++i) {
    TINY_LLM_CHECK(tiny_llm::InvalidArgumentError,
                   valid_size.data<uint32_t>()[i] <= dim);
    TINY_LLM_CHECK(tiny_llm::InvalidArgumentError,
                   valid_size.data<uint32_t>()[i] >= min_tokens_to_keep_);
  }

  for (int64_t b = 0; b < batch; ++b) {
    auto shift = b * dim;
    auto *logit_ptr = logit.data<float>() + shift;
    auto logit_size = valid_size.data<uint32_t>()[b];
    quick_sort(logit_ptr, id.data<uint32_t>() + shift, 0,
               static_cast<int32_t>(logit_size) - 1);

    std::vector<float> tmp(logit_size);
    float exp_sum{};
    for (uint32_t i = 0; i < logit_size; ++i) {
      tmp.at(i) = std::exp(logit_ptr[i] - logit_ptr[0]);
      exp_sum += tmp.at(i);
    }

    auto remain_p = 1 - top_p_;
    auto valid_idx = logit_size - 1;
    for (; valid_idx >= min_tokens_to_keep_; --valid_idx) {
      remain_p -= tmp.at(valid_idx) / exp_sum;
      if (remain_p < 0) {
        break;
      }
    }
    valid_size.data<uint32_t>()[b] = valid_idx + 1;
  }
}

static_assert(LogitProcessor<TopPProcessor>);
} // namespace tiny_llm
