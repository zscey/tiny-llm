#include "tiny_llm/logit_processors/argmax_processor.hpp"
#include "tiny_llm/common/log_and_excepts.hpp"

namespace tiny_llm {
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void ArgmaxProcessor::apply(Tensor &logit, Tensor &id,
                            Tensor &valid_size) const {
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

  for (int64_t i = 0; i < batch; ++i) {
    auto shift = i * dim;
    auto *logit_ptr = logit.data<float>() + shift;
    auto *id_ptr = id.data<uint32_t>() + shift;
    auto logit_size = valid_size.data<uint32_t>()[i];

    uint32_t max_id{};
    for (uint32_t j = 1; j < logit_size; ++j) {
      if (logit_ptr[j] > logit_ptr[max_id]) {
        max_id = j;
      }
    }
    std::swap(logit_ptr[0], logit_ptr[max_id]);
    std::swap(id_ptr[0], id_ptr[max_id]);

    valid_size.data<uint32_t>()[i] = 1;
  }
}

static_assert(LogitProcessor<ArgmaxProcessor>);
} // namespace tiny_llm
