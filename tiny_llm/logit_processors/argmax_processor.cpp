#include "tiny_llm/logit_processors/argmax_processor.hpp"
#include "tiny_llm/common/checks.hpp"
#include "tiny_llm/common/exception.hpp"

namespace tiny_llm {
void ArgmaxProcessor::apply(Tensor &logit, Tensor &id,
                            Tensor &valid_size) const {
  (void)this;
  check_params(logit, id, valid_size);

  int64_t batch = logit.shape().at(0);
  int64_t dim = logit.shape().at(1);
  for (int64_t i = 0; i < batch; ++i) {
    TINY_LLM_CHECK(InvalidArgumentError, valid_size.data<uint32_t>()[i] <= dim);
    TINY_LLM_CHECK(InvalidArgumentError, valid_size.data<uint32_t>()[i] >= 1);
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
