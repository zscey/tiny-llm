#include "tiny_llm/logit_processors/sample_processors.hpp"
#include "tiny_llm/common/log_and_excepts.hpp"
#include <random>

namespace tiny_llm {
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void MultinomialProcessor::apply(Tensor &logit, Tensor &id,
                                 Tensor &valid_size) const {
  thread_local std::mt19937 engine(std::random_device{}());
  thread_local std::uniform_real_distribution<> dist(0.F, 1.F);

  (void)this;
  TINY_LLM_CHECK(logit.dtype() == DataType::kFloat32);
  TINY_LLM_CHECK(id.dtype() == DataType::kUint32);
  TINY_LLM_CHECK(valid_size.dtype() == DataType::kUint32);
  TINY_LLM_CHECK(logit.shape().size() == 2);
  TINY_LLM_CHECK(logit.shape() == id.shape());

  int64_t batch = logit.shape().at(0);
  int64_t dim = logit.shape().at(1);
  TINY_LLM_CHECK(valid_size.shape() == std::vector<int64_t>{batch})
  for (int64_t i = 0; i < batch; ++i) {
    TINY_LLM_CHECK(valid_size.data<uint32_t>()[i] <= dim);
    TINY_LLM_CHECK(valid_size.data<uint32_t>()[i] >= 1);
  }

  for (int64_t b = 0; b < batch; ++b) {
    auto shift = b * dim;
    auto *logit_ptr = logit.data<float>() + shift;
    auto *id_ptr = id.data<uint32_t>() + shift;
    auto logit_size = valid_size.data<uint32_t>()[b];

    float max_logit{std::numeric_limits<float>::lowest()};
    for (uint32_t i = 0; i < logit_size; ++i) {
      max_logit = std::max(max_logit, logit_ptr[i]);
    }
    float exp_sum{};
    for (uint32_t i = 0; i < logit_size; ++i) {
      logit_ptr[i] = std::exp(logit_ptr[i] - max_logit);
      exp_sum += logit_ptr[i];
    }

    auto random_number = dist(engine);
    uint32_t sampled_id{};
    for (; sampled_id < logit_size; ++sampled_id) {
      random_number -= logit_ptr[sampled_id] / exp_sum;
      if (random_number < 0) {
        break;
      }
    }
    sampled_id = std::min(sampled_id, logit_size - 1);
    std::swap(logit_ptr[0], logit_ptr[sampled_id]);
    std::swap(id_ptr[0], id_ptr[sampled_id]);

    valid_size.data<uint32_t>()[b] = 1;
  }
}

static_assert(LogitProcessor<MultinomialProcessor>);
} // namespace tiny_llm
