#include "tiny_llm/graph/graph.hpp"
#include "tiny_llm/common/log_and_excepts.hpp"

namespace tiny_llm {

namespace {
auto get_tensor(Graph &graph, const std::string &name) -> TensorInfo & {
  if (auto iter = graph.tensor_name_to_idx.find(name);
      iter != graph.tensor_name_to_idx.end()) {
    auto &cur_tensor_info = graph.tensor_infos[iter->second];
    TINY_LLM_CHECK(cur_tensor_info);
    return cur_tensor_info.value();
  }

  graph.tensor_infos.emplace_back(TensorInfo{});
  graph.tensor_name_to_idx.emplace(name, graph.tensor_infos.size() - 1);
  return graph.tensor_infos.back().value();
}
} // namespace

void Graph::add_tensor(const std::string &name, DataType dtype,
                       std::vector<int64_t> shape) {
  auto &cur_tensor_info = get_tensor(*this, name);
  TINY_LLM_CHECK(cur_tensor_info.is_initialized == false);

  cur_tensor_info.is_initialized = true;
  cur_tensor_info.dtype = dtype;
  cur_tensor_info.shape = std::move(shape);
}

void Graph::add_node(const std::string &name, const NodeIONames &node_io_names,
                     Param param) {
  TINY_LLM_CHECK(!node_name_to_idx.contains(name));

  // TODO(): construct by move
  auto &cur_node = nodes.emplace_back(Node{.param = param});
  auto cur_id = nodes.size() - 1;
  node_name_to_idx.emplace(name, cur_id);

  for (const auto &name : node_io_names.input_names) {
    auto &cur_tensor_info = get_tensor(*this, name);
    cur_tensor_info.consumer_nodes.emplace_back(cur_id);
    cur_node->input_tensors.emplace_back(tensor_name_to_idx.at(name));
  }

  for (const auto &name : node_io_names.output_names) {
    auto &cur_tensor_info = get_tensor(*this, name);
    TINY_LLM_CHECK(!cur_tensor_info.producer_node);
    cur_tensor_info.producer_node = cur_id;
    cur_node->output_tensors.emplace_back(tensor_name_to_idx.at(name));
  }
}

void Graph::set_input_names(std::vector<std::string> input_names) {
  for (const auto &name : input_names) {
    TINY_LLM_CHECK(tensor_name_to_idx.contains(name));
  }

  this->input_names = std::move(input_names);
}

void Graph::set_output_names(std::vector<std::string> output_names) {
  for (const auto &name : output_names) {
    TINY_LLM_CHECK(tensor_name_to_idx.contains(name));
  }

  this->output_names = std::move(output_names);
}
} // namespace tiny_llm
