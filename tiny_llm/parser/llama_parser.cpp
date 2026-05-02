#include "tiny_llm/parser/llama_parser.hpp"
#include "tiny_llm/common/checks.hpp"
#include "tiny_llm/common/exception.hpp"
#include <array>

namespace tiny_llm {
namespace {
template <typename T>
auto llama_add_node(
    Graph &g, const std::string &node_name, const T &param,
    std::vector<std::string>
        extern_inputs) // NOLINT(performance-unnecessary-value-param)
    -> std::array<std::string, output_num<T>()> {
  TINY_LLM_CHECK(tiny_llm::InvalidArgumentError,
                 extern_inputs.size() == extern_input_num<T>());
  TINY_LLM_CHECK(tiny_llm::InvalidArgumentError, output_num<T>() == 1);
  std::array<std::string, output_num<T>()> output_names{node_name + ".out"};

  g.add_node(node_name,
             {.input_names = std::move(extern_inputs),
              .output_names = {output_names.begin(), output_names.end()}},
             param);
  return output_names;
}

auto llama_add_node(Graph &g, const std::string &node_name,
                    const EmbeddingParam &param,
                    std::vector<std::string> extern_inputs)
    -> std::array<std::string, output_num<EmbeddingParam>()> {
  TINY_LLM_CHECK(tiny_llm::InvalidArgumentError,
                 extern_inputs.size() == extern_input_num<EmbeddingParam>());
  TINY_LLM_CHECK(tiny_llm::InvalidArgumentError,
                 output_num<EmbeddingParam>() == 1);
  std::array<std::string, output_num<EmbeddingParam>()> output_names{node_name +
                                                                     ".out"};

  g.add_tensor(node_name + ".weight", DataType::kFloat32,
               {param.num_embeddings, param.hidden_size});

  extern_inputs.emplace_back(node_name + ".weight");
  g.add_node(node_name,
             {.input_names = std::move(extern_inputs),
              .output_names = {output_names.begin(), output_names.end()}},
             param);
  return output_names;
}

auto llama_add_node(Graph &g, const std::string &node_name,
                    const RopeParam &param,
                    const std::vector<std::string> &extern_inputs)
    -> std::array<std::string, output_num<RopeParam>()> {
  TINY_LLM_CHECK(tiny_llm::InvalidArgumentError,
                 extern_inputs.size() == extern_input_num<RopeParam>());
  TINY_LLM_CHECK(tiny_llm::InvalidArgumentError, output_num<RopeParam>() == 2);
  std::array<std::string, output_num<RopeParam>()> output_names{
      node_name + ".cos", node_name + ".sin"};

  g.add_node(node_name,
             {.output_names = {output_names.begin(), output_names.end()}},
             param);
  return output_names;
}

auto llama_add_node(Graph &g, const std::string &node_name,
                    const RMSNormParam &param,
                    std::vector<std::string> extern_inputs)
    -> std::array<std::string, output_num<RMSNormParam>()> {
  TINY_LLM_CHECK(tiny_llm::InvalidArgumentError,
                 extern_inputs.size() == extern_input_num<RMSNormParam>());
  TINY_LLM_CHECK(tiny_llm::InvalidArgumentError,
                 output_num<RMSNormParam>() == 1);
  std::array<std::string, output_num<RMSNormParam>()> output_names{node_name +
                                                                   ".out"};

  g.add_tensor(node_name + ".weight", DataType::kFloat32, {param.hidden_size});

  extern_inputs.emplace_back(node_name + ".weight");
  g.add_node(node_name,
             {.input_names = std::move(extern_inputs),
              .output_names = {output_names.begin(), output_names.end()}},
             param);
  return output_names;
}

auto llama_add_node(Graph &g, const std::string &node_name,
                    const LinearParam &param,
                    std::vector<std::string> extern_inputs)
    -> std::array<std::string, output_num<LinearParam>()> {
  TINY_LLM_CHECK(tiny_llm::InvalidArgumentError,
                 extern_inputs.size() == extern_input_num<LinearParam>());
  TINY_LLM_CHECK(tiny_llm::InvalidArgumentError, param.bias == false);
  TINY_LLM_CHECK(tiny_llm::InvalidArgumentError,
                 output_num<LinearParam>() == 1);
  std::array<std::string, output_num<LinearParam>()> output_names{node_name +
                                                                  ".out"};

  g.add_tensor(node_name + ".weight", DataType::kFloat32,
               {param.out_dim, param.in_dim});

  extern_inputs.emplace_back(node_name + ".weight");
  g.add_node(node_name,
             {.input_names = std::move(extern_inputs),
              .output_names = {output_names.begin(), output_names.end()}},
             param);
  return output_names;
}

auto llama_add_node(Graph &g, const std::string &node_name,
                    const CausalAttentionParam &param,
                    std::vector<std::string> extern_inputs)
    -> std::array<std::string, output_num<CausalAttentionParam>()> {
  // hidden_state, cos, sin, pos_ids
  TINY_LLM_CHECK(tiny_llm::InvalidArgumentError,
                 extern_inputs.size() ==
                     extern_input_num<CausalAttentionParam>());
  TINY_LLM_CHECK(tiny_llm::InvalidArgumentError, param.bias == false);
  TINY_LLM_CHECK(tiny_llm::InvalidArgumentError,
                 output_num<CausalAttentionParam>() == 1);
  std::array<std::string, output_num<CausalAttentionParam>()> output_names{
      node_name + ".out"};

  auto head_dim = static_cast<int64_t>(param.head_dim);
  auto q_dim = param.q_head * head_dim;
  auto kv_dim = param.kv_head * head_dim;
  g.add_tensor(node_name + ".q_proj.weight", DataType::kFloat32,
               {q_dim, q_dim});
  g.add_tensor(node_name + ".k_proj.weight", DataType::kFloat32,
               {kv_dim, q_dim});
  g.add_tensor(node_name + ".v_proj.weight", DataType::kFloat32,
               {kv_dim, q_dim});
  g.add_tensor(node_name + ".o_proj.weight", DataType::kFloat32,
               {q_dim, q_dim});

  extern_inputs.emplace_back(node_name + ".q_proj.weight");
  extern_inputs.emplace_back(node_name + ".k_proj.weight");
  extern_inputs.emplace_back(node_name + ".v_proj.weight");
  extern_inputs.emplace_back(node_name + ".o_proj.weight");
  g.add_node(node_name,
             {.input_names = std::move(extern_inputs),
              .output_names = {output_names.begin(), output_names.end()}},
             param);
  return output_names;
}

auto llama_add_node(Graph &g, const std::string &node_name,
                    const SliceLinearParam &param,
                    std::vector<std::string> extern_inputs)
    -> std::array<std::string, output_num<SliceLinearParam>()> {
  TINY_LLM_CHECK(tiny_llm::InvalidArgumentError,
                 extern_inputs.size() == extern_input_num<SliceLinearParam>());
  TINY_LLM_CHECK(tiny_llm::InvalidArgumentError, param.bias == false);
  TINY_LLM_CHECK(tiny_llm::InvalidArgumentError,
                 output_num<SliceLinearParam>() == 1);
  std::array<std::string, output_num<SliceLinearParam>()> output_names{
      node_name + ".out"};

  g.add_tensor(node_name + ".weight", DataType::kFloat32,
               {param.out_dim, param.in_dim});

  extern_inputs.emplace_back(node_name + ".weight");
  g.add_node(node_name,
             {.input_names = std::move(extern_inputs),
              .output_names = {output_names.begin(), output_names.end()}},
             param);
  return output_names;
}
} // namespace

auto llama_parser(const nlohmann::json &config) -> Graph {
  TINY_LLM_CHECK(tiny_llm::InvalidArgumentError,
                 config["attention_bias"].get<bool>() == false);
  TINY_LLM_CHECK(tiny_llm::InvalidArgumentError,
                 config["rope_scaling"].is_null());
  TINY_LLM_CHECK(tiny_llm::InvalidArgumentError,
                 config["hidden_act"].get<std::string>() == "silu");
  auto hidden_size = config["hidden_size"].get<uint32_t>();
  auto q_head = config["num_attention_heads"].get<uint32_t>();
  auto kv_head = config["num_key_value_heads"].get<uint32_t>();
  auto head_dim = hidden_size / q_head;

  Graph g;
  std::string input_token_name{"token_ids"};
  g.add_tensor(input_token_name, DataType::kUint32, {1, 2});
  std::string input_pos_ids_name{"pos_ids"};
  g.add_tensor(input_pos_ids_name, DataType::kUint32, {1, 2});

  auto [embed_node_out] = llama_add_node(
      g, "model.embed_tokens",
      EmbeddingParam{.num_embeddings = config["vocab_size"].get<uint32_t>(),
                     .hidden_size = hidden_size},
      {input_token_name});
  auto [rope_cos, rope_sin] = llama_add_node(
      g, "rope",
      RopeParam{.max_len = config["max_position_embeddings"].get<uint32_t>(),
                .head_dim = head_dim,
                .theta = config["rope_theta"].get<double>(),
                .pin = true},
      {});

  std::string last_hidden_state = embed_node_out;
  for (uint32_t layer_id = 0,
                layer_num = config["num_hidden_layers"].get<uint32_t>();
       layer_id < layer_num; ++layer_id) {
    auto [input_norm_out] = llama_add_node(
        g, fmt::format("model.layers.{}.input_layernorm", layer_id),
        RMSNormParam{.hidden_size = hidden_size,
                     .eps = config["rms_norm_eps"].get<float>(),
                     .inplace = false},
        {last_hidden_state});
    auto [attn_out] = llama_add_node(
        g, fmt::format("model.layers.{}.self_attn", layer_id),
        CausalAttentionParam{
            .head_dim = head_dim,
            .q_head = q_head,
            .kv_head = kv_head,
            .bias = false,
            .max_len = config["max_position_embeddings"].get<uint32_t>()},
        {input_norm_out, rope_cos, rope_sin, input_pos_ids_name});
    auto [attn_residual] = llama_add_node(
        g, fmt::format("model.layers.{}.attn_residual", layer_id),
        AddParam{.left_idx = 0, .right_idx = 1, .out_idx = 1},
        {last_hidden_state, attn_out});

    auto [mlp_norm] = llama_add_node(
        g, fmt::format("model.layers.{}.post_attention_layernorm", layer_id),
        RMSNormParam{.hidden_size = hidden_size,
                     .eps = config["rms_norm_eps"].get<float>(),
                     .inplace = false},
        {attn_residual});
    auto [gate_out] = llama_add_node(
        g, fmt::format("model.layers.{}.mlp.gate_proj", layer_id),
        LinearParam{.in_dim = hidden_size,
                    .out_dim = config["intermediate_size"].get<uint32_t>(),
                    .bias = false},
        {mlp_norm});
    auto [act_out] =
        llama_add_node(g, fmt::format("model.layers.{}.mlp.act", layer_id),
                       SiLUParam{.inplace = true}, {gate_out});
    auto [up_out] = llama_add_node(
        g, fmt::format("model.layers.{}.mlp.up_proj", layer_id),
        LinearParam{.in_dim = hidden_size,
                    .out_dim = config["intermediate_size"].get<uint32_t>(),
                    .bias = false},
        {mlp_norm});
    auto [mul_out] =
        llama_add_node(g, fmt::format("model.layers.{}.mlp.mul", layer_id),
                       MulParam{.left_idx = 0, .right_idx = 1, .out_idx = 1},
                       {act_out, up_out});
    auto [mlp_out] = llama_add_node(
        g, fmt::format("model.layers.{}.mlp.down_proj", layer_id),
        LinearParam{.in_dim = config["intermediate_size"].get<uint32_t>(),
                    .out_dim = hidden_size,
                    .bias = false},
        {mul_out});

    auto [mlp_residual] =
        llama_add_node(g, fmt::format("model.layers.{}.mlp_residual", layer_id),
                       AddParam{.left_idx = 0, .right_idx = 1, .out_idx = 1},
                       {attn_residual, mlp_out});
    last_hidden_state = mlp_residual;
  }

  auto [model_norm] =
      llama_add_node(g, "model.norm",
                     RMSNormParam{.hidden_size = hidden_size,
                                  .eps = config["rms_norm_eps"].get<float>(),
                                  .inplace = true},
                     {last_hidden_state});
  auto [lm_head] = llama_add_node(
      g, "lm_head",
      SliceLinearParam{.in_dim = hidden_size,
                       .out_dim = config["vocab_size"].get<uint32_t>(),
                       .bias = false,
                       .only_last_q = 1},
      {model_norm});

  g.set_input_names({input_token_name, input_pos_ids_name});
  g.set_output_names({lm_head});
  return g;
}
} // namespace tiny_llm
