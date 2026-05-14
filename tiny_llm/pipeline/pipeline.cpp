#include "tiny_llm/pipeline/pipeline.hpp"
#include "tiny_llm/common/checks.hpp"
#include "tiny_llm/common/exception.hpp"
#include "tiny_llm/logit_processors/argmax_processor.hpp"
#include "tiny_llm/logit_processors/sample_processors.hpp"
#include "tiny_llm/logit_processors/sort_processors.hpp"
#include "tiny_llm/logit_processors/temperature_processor.hpp"
#include "tiny_llm/parser/llama_parser.hpp"
#include "tiny_llm/runtime/cuda/cuda_runtime.hpp"
#include "tiny_llm/tokenizers/huggingface/tokenizers.hpp"
#include "tiny_llm/weight_managers/safetensors/weight_manager.hpp"
#include <filesystem>
#include <fstream>

namespace tiny_llm {
namespace {
auto all_zero(const Tensor &unfinished) -> bool {
  const auto *cur_ptr = unfinished.data<uint32_t>();
  for (int64_t b = 0, b_end = unfinished.shape().at(0); b < b_end; ++b) {
    if (cur_ptr[b] != 0) {
      return false;
    }
  }
  return true;
}
} // namespace

Pipeline::Pipeline(const std::string &model_path, PipelineConfig config) {
  TINY_LLM_CHECK(InvalidArgumentError,
                 config.model_type == ModelType::kTinyLlama);
  TINY_LLM_CHECK(InvalidArgumentError, config.dtype == DataType::kFloat32);
  TINY_LLM_CHECK(InvalidArgumentError, config.device.type == DeviceType::kCuda);
  // TODO(): support batch > 1
  TINY_LLM_CHECK(InvalidArgumentError, config.batch == 1);

  std::filesystem::path model_root(model_path);

  // tokenizer_
  tokenizer_ = std::make_unique<TokenizerWrapper>(
      HuggingfaceTokenizer(model_root / "tokenizer.json"));

  // runtime_
  std::ifstream f(model_root / "config.json");
  auto json = nlohmann::json::parse(f);
  TINY_LLM_CHECK(InvalidArgumentError,
                 (!json["torch_dtype"].is_null() &&
                  json["torch_dtype"].get<std::string>() == "float32"));
  f.close();
  auto plan = cuda::create_cuda_plan(
      llama_parser(json),
      {.named_shape_ranges = {
           {"token_ids",
            {.min_shape = {config.batch, 1},
             .max_shape = {config.batch,
                           json["max_position_embeddings"].get<uint32_t>()}}},
           {"pos_ids",
            {.min_shape = {config.batch, 1},
             .max_shape = {config.batch, json["max_position_embeddings"]
                                             .get<uint32_t>()}}}}});
  WeightManagerWrapper wm(
      SafeTensorWeightManager(model_root / "model.safetensors"));
  runtime_ = std::make_unique<cuda::CudaRuntime>(std::move(plan), wm);

  // output_
  output_ = std::make_unique<Tensor>(
      Device{.type = DeviceType::kCpu, .id = 0}, DataType::kFloat32,
      std::vector<int64_t>{static_cast<int64_t>(config.batch), 1,
                           json["vocab_size"].get<int64_t>()});

  // eos_token_id_
  eos_token_id_ = json["eos_token_id"].is_null()
                      ? json["vocab_size"].get<uint32_t>()
                      : json["eos_token_id"].get<uint32_t>();

  // indexes_
  indexes_ = std::make_unique<Tensor>(
      Device{.type = DeviceType::kCpu, .id = 0}, DataType::kUint32,
      std::vector<int64_t>{static_cast<int64_t>(config.batch),
                           json["vocab_size"].get<int64_t>()});

  // valid_size_
  valid_size_ = std::make_unique<Tensor>(
      indexes_->device(), DataType::kUint32,
      std::vector<int64_t>{static_cast<int64_t>(config.batch)});

  // unfinished_
  unfinished_ = std::make_unique<Tensor>(
      valid_size_->device(), valid_size_->dtype(), valid_size_->shape());
}

auto Pipeline::apply(const std::vector<std::string> &prompts,
                     uint32_t max_new_tokens, bool do_sample, float temperature,
                     uint32_t top_k, float top_p) const
    -> std::vector<std::string> {
  // TODO(): support batch > 1
  TINY_LLM_CHECK(InvalidArgumentError,
                 prompts.size() ==
                     static_cast<size_t>(unfinished_->shape().at(0)));

  auto tokenize_res = tokenizer_->encode(prompts.at(0), true);
  {
    runtime_->set_prefill(true);

    Tensor token_ids(
        {.type = DeviceType::kCpu, .id = 0}, DataType::kUint32,
        {1, static_cast<int64_t>(tokenize_res.size())}, {}, 0,
        std::make_shared<Buffer>(tokenize_res.data(),
                                 tokenize_res.size() * sizeof(uint32_t),
                                 Device{.type = DeviceType::kCpu, .id = 0}));
    Tensor pos_ids({.type = DeviceType::kCpu, .id = 0}, DataType::kUint32,
                   {1, static_cast<int64_t>(tokenize_res.size())});
    auto *pos_ids_ptr = pos_ids.data<uint32_t>();
    std::iota(pos_ids_ptr, pos_ids_ptr + tokenize_res.size(), 0);

    runtime_->cpu_tensor_copy_to_input("token_ids", token_ids);
    runtime_->cpu_tensor_copy_to_input("pos_ids", pos_ids);
    runtime_->execute();
    runtime_->output_copy_to_cpu_tensor("lm_head.out", *output_);
  }

  std::vector<uint32_t> generated_tokens;
  generated_tokens.reserve(max_new_tokens);
  {
    bool from_prefill{true};
    runtime_->set_prefill(false);

    std::vector<LogitProcessorWrapper> wrappers;
    if (do_sample) {
      wrappers.emplace_back(TemperatureProcessor{temperature});
      wrappers.emplace_back(TopKProcessor{top_k});
      wrappers.emplace_back(TopPProcessor{top_p});
      wrappers.emplace_back(MultinomialProcessor{});
    } else {
      wrappers.emplace_back(ArgmaxProcessor{});
    }

    auto vocab_size = static_cast<uint32_t>(output_->shape().back());
    std::vector<int64_t> target_shape{1, vocab_size};
    *unfinished_->data<uint32_t>() = 1;
    Tensor token_ids({.type = DeviceType::kCpu, .id = 0}, DataType::kUint32,
                     {1, 1});
    Tensor pos_ids({.type = DeviceType::kCpu, .id = 0}, DataType::kUint32,
                   {1, 1});

    while (generated_tokens.size() < max_new_tokens &&
           !all_zero(*unfinished_)) {
      if (!from_prefill) {
        runtime_->cpu_tensor_copy_to_input("token_ids", token_ids);
        runtime_->cpu_tensor_copy_to_input("pos_ids", pos_ids);
        runtime_->execute();
        runtime_->output_copy_to_cpu_tensor("lm_head.out", *output_);
      }

      output_->reshape(target_shape);
      std::iota(indexes_->data<uint32_t>(),
                indexes_->data<uint32_t>() + vocab_size, 0);
      *valid_size_->data<uint32_t>() = vocab_size;

      for (auto &processor : wrappers) {
        processor.apply(*output_, *indexes_, *valid_size_);
      }
      generated_tokens.emplace_back(*indexes_->data<uint32_t>());
      *unfinished_->data<uint32_t>() =
          static_cast<uint32_t>(generated_tokens.back() != eos_token_id_);

      from_prefill = false;
      *token_ids.data<uint32_t>() = generated_tokens.back();
      *pos_ids.data<uint32_t>() =
          tokenize_res.size() + generated_tokens.size() - 1;
    }
  }

  return {tokenizer_->decode(generated_tokens, true)};
}
} // namespace tiny_llm
