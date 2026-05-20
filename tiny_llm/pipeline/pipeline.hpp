#pragma once

#include "tiny_llm/runtime/iruntime.hpp"
#include "tiny_llm/tensor/tensor.hpp"
#include "tiny_llm/tokenizers/tokenizers.hpp"

namespace tiny_llm {
enum class ModelType : std::uint8_t {
  kTinyLlama,
};

class Pipeline {
public:
  struct PipelineConfig {
    ModelType model_type{};
    DataType dtype{};
    Device device{};

    bool paged{false};
    uint32_t max_requests{};
  };

  Pipeline(const std::string &model_path, PipelineConfig config);

  [[nodiscard]] auto apply(const std::vector<std::string> &prompts,
                           uint32_t max_new_tokens = 256, bool do_sample = true,
                           float temperature = 0.7F, uint32_t top_k = 50,
                           float top_p = 0.95F) const
      -> std::vector<std::string>;

private:
  bool paged_;
  std::unique_ptr<TokenizerWrapper> tokenizer_;
  std::unique_ptr<IRuntime> runtime_;
  std::unique_ptr<Tensor> output_;

  uint32_t eos_token_id_;
  std::unique_ptr<Tensor> indexes_;
  std::unique_ptr<Tensor> valid_size_;
  std::unique_ptr<Tensor> unfinished_;
};
} // namespace tiny_llm
