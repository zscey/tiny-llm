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

Pipeline::Pipeline(const std::string &model_path, PipelineConfig config)
    : paged_(config.paged), max_requests_(config.max_requests) {
  TINY_LLM_CHECK(InvalidArgumentError,
                 config.model_type == ModelType::kTinyLlama);
  TINY_LLM_CHECK(InvalidArgumentError, config.dtype == DataType::kFloat32);
  TINY_LLM_CHECK(InvalidArgumentError, config.device.type == DeviceType::kCuda);
  if (!config.paged) {
    TINY_LLM_CHECK(InvalidArgumentError, config.max_requests == 1);
  }

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
            {.min_shape = {1, 1},
             .max_shape = {1,
                           static_cast<uint32_t>(config.max_requests *
                                                 json["max_position_embeddings"]
                                                     .get<uint32_t>())}}},
           {"pos_ids",
            {.min_shape = {1, 1},
             .max_shape = {
                 1, static_cast<uint32_t>(
                        config.max_requests *
                        json["max_position_embeddings"].get<uint32_t>())}}}}});
  // TODO(hao.lin): unify runtime creation
  WeightManagerWrapper wm(
      SafeTensorWeightManager(model_root / "model.safetensors"));
  runtime_ = std::make_unique<cuda::CudaRuntime>(std::move(plan), wm);

  // output_
  output_ = std::make_unique<Tensor>(
      Device{.type = DeviceType::kCpu, .id = 0}, DataType::kFloat32,
      std::vector<int64_t>{static_cast<int64_t>(config.max_requests), 1,
                           json["vocab_size"].get<int64_t>()});

  // eos_token_id_
  eos_token_id_ = json["eos_token_id"].is_null()
                      ? json["vocab_size"].get<uint32_t>()
                      : json["eos_token_id"].get<uint32_t>();

  // indexes_
  indexes_ = std::make_unique<Tensor>(
      Device{.type = DeviceType::kCpu, .id = 0}, DataType::kUint32,
      std::vector<int64_t>{static_cast<int64_t>(config.max_requests),
                           json["vocab_size"].get<int64_t>()});

  // valid_size_
  valid_size_ = std::make_unique<Tensor>(
      indexes_->device(), DataType::kUint32,
      std::vector<int64_t>{static_cast<int64_t>(config.max_requests)});

  // unfinished_
  unfinished_ = std::make_unique<Tensor>(
      valid_size_->device(), valid_size_->dtype(), valid_size_->shape());
}

auto Pipeline::apply(const std::vector<std::string> &prompts,
                     uint32_t max_new_tokens, bool do_sample, float temperature,
                     uint32_t top_k, float top_p) const
    -> std::vector<std::string> {
  if (!paged_) {
    TINY_LLM_CHECK(InvalidArgumentError, prompts.size() == 1);
  }

  uint32_t total_queries{};
  std::vector<std::vector<uint32_t>> tokenize_res;
  tokenize_res.reserve(prompts.size());
  for (const auto &prompt : prompts) {
    tokenize_res.emplace_back(tokenizer_->encode(prompt, true));
    total_queries += static_cast<uint32_t>(tokenize_res.back().size());
  }
  {
    Tensor token_ids({.type = DeviceType::kCpu, .id = 0}, DataType::kUint32,
                     {1, static_cast<int64_t>(total_queries)});
    Tensor pos_ids({.type = DeviceType::kCpu, .id = 0}, DataType::kUint32,
                   {1, static_cast<int64_t>(total_queries)});
    auto *token_ids_ptr = token_ids.data<uint32_t>();
    auto *pos_ids_ptr = pos_ids.data<uint32_t>();
    for (const auto &r : tokenize_res) {
      std::memcpy(token_ids_ptr, r.data(), r.size() * sizeof(uint32_t));
      token_ids_ptr += r.size();

      std::iota(pos_ids_ptr, pos_ids_ptr + r.size(), 0);
      pos_ids_ptr += r.size();
    }

    // TODO(hao.lin): call runtime_->set_meta in paged case
    runtime_->set_prefill(true);

    runtime_->cpu_tensor_copy_to_input("token_ids", token_ids);
    runtime_->cpu_tensor_copy_to_input("pos_ids", pos_ids);
    runtime_->execute();
    runtime_->output_copy_to_cpu_tensor("lm_head.out", *output_);
  }

  std::vector<std::vector<uint32_t>> generated_tokens(prompts.size());
  for (auto &r : generated_tokens) {
    r.reserve(max_new_tokens);
  }
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

    auto prompt_size = static_cast<int64_t>(prompts.size());
    auto vocab_size = output_->shape().back();
    std::vector<int64_t> target_shape{prompt_size, vocab_size};
    Tensor token_ids({.type = DeviceType::kCpu, .id = 0}, DataType::kUint32,
                     {prompt_size, 1});
    Tensor pos_ids({.type = DeviceType::kCpu, .id = 0}, DataType::kUint32,
                   {prompt_size, 1});

    auto *unfinished_ptr = unfinished_->data<uint32_t>();
    for (int64_t i = 0; i < prompt_size; ++i) {
      unfinished_ptr[i] = 1;
    }
    while (generated_tokens[0].size() < max_new_tokens &&
           !all_zero(*unfinished_)) {
      if (!from_prefill) {
        runtime_->cpu_tensor_copy_to_input("token_ids", token_ids);
        runtime_->cpu_tensor_copy_to_input("pos_ids", pos_ids);
        runtime_->execute();
        runtime_->output_copy_to_cpu_tensor("lm_head.out", *output_);
      }

      output_->reshape(target_shape);

      {
        auto *indexes_ptr = indexes_->data<uint32_t>();
        auto *valid_size_ptr = valid_size_->data<uint32_t>();
        for (int64_t i = 0; i < prompt_size; ++i) {
          auto *cur_indexes_ptr = indexes_ptr + (i * vocab_size);
          std::iota(cur_indexes_ptr, cur_indexes_ptr + vocab_size, 0);
          valid_size_ptr[i] = vocab_size;
        }
      }

      for (auto &processor : wrappers) {
        processor.apply(*output_, *indexes_, *valid_size_);
      }

      from_prefill = false;

      {
        const auto *indexes_ptr = indexes_->data<uint32_t>();
        auto *unfinished_ptr = unfinished_->data<uint32_t>();
        auto *token_ids_ptr = token_ids.data<uint32_t>();
        auto *pos_ids_ptr = pos_ids.data<uint32_t>();
        for (int64_t i = 0; i < prompt_size; ++i) {
          auto next_token = (unfinished_ptr[i] == 1)
                                ? indexes_ptr[i * vocab_size]
                                : eos_token_id_;
          unfinished_ptr[i] =
              static_cast<uint32_t>(next_token != eos_token_id_);
          generated_tokens.at(i).emplace_back(next_token);
          token_ids_ptr[i] = next_token;
          pos_ids_ptr[i] =
              tokenize_res.at(i).size() + generated_tokens.at(i).size() - 1;
        }
      }
    }
  }

  std::vector<std::string> res;
  res.reserve(generated_tokens.size());
  for (const auto &tokens : generated_tokens) {
    res.emplace_back(tokenizer_->decode(tokens, true));
  }
  return res;
}
} // namespace tiny_llm
