#include "tiny_llm/graph/graph.hpp"
#include "tiny_llm/common/log_and_excepts.hpp"
#include <algorithm>
#include <numeric>
#include <queue>

namespace tiny_llm {

namespace {
auto get_tensor(Graph &graph, const std::string &name) -> TensorInfo & {
  if (auto iter = graph.tensor_name_to_idx.find(name);
      iter != graph.tensor_name_to_idx.end()) {
    auto &cur_tensor_info = graph.tensor_infos[iter->second];
    TINY_LLM_CHECK(cur_tensor_info);
    return cur_tensor_info.value().second;
  }

  graph.tensor_infos.emplace_back(std::in_place, name, TensorInfo{});
  graph.tensor_name_to_idx.emplace(name, graph.tensor_infos.size() - 1);
  return graph.tensor_infos.back().value().second;
}
} // namespace

void Graph::add_tensor(const std::string &name, DataType dtype,
                       std::vector<int64_t> shape) {
  auto &cur_tensor_info = get_tensor(*this, name);
  TINY_LLM_CHECK(cur_tensor_info.has_explicit_added == false);

  cur_tensor_info.has_explicit_added = true;
  cur_tensor_info.dtype = dtype;
  cur_tensor_info.shape = std::move(shape);
}

void Graph::add_node(const std::string &name, const NodeIONames &node_io_names,
                     Param param) {
  TINY_LLM_CHECK(!node_name_to_idx.contains(name));

  // TODO(): construct by move
  auto &cur_node =
      nodes.emplace_back(std::in_place, name, Node{.param = param});
  auto cur_id = nodes.size() - 1;
  node_name_to_idx.emplace(name, cur_id);

  for (const auto &name : node_io_names.input_names) {
    auto &cur_tensor_info = get_tensor(*this, name);
    cur_tensor_info.consumer_nodes.emplace_back(cur_id);
    cur_node->second.input_tensors.emplace_back(tensor_name_to_idx.at(name));
  }

  for (const auto &name : node_io_names.output_names) {
    auto &cur_tensor_info = get_tensor(*this, name);
    TINY_LLM_CHECK(!cur_tensor_info.producer_node);
    cur_tensor_info.producer_node = cur_id;
    cur_node->second.output_tensors.emplace_back(tensor_name_to_idx.at(name));
  }
}

void Graph::set_input_names(std::vector<std::string> input_names) {
  TINY_LLM_CHECK(std::ranges::all_of(input_names, [this](const auto &name) {
    return tensor_name_to_idx.contains(name);
  }))

  this->input_names = std::move(input_names);
}

void Graph::set_output_names(std::vector<std::string> output_names) {
  TINY_LLM_CHECK(std::ranges::all_of(output_names, [this](const auto &name) {
    return tensor_name_to_idx.contains(name);
  }))

  this->output_names = std::move(output_names);
}

namespace {
auto is_valid_map(const std::unordered_map<std::string, uint32_t> &map)
    -> bool {
  auto map_size = map.size();

  std::vector<uint32_t> exist(map_size, 0);
  for (const auto &[_, id] : map) {
    if (id >= map_size) {
      return false;
    }
    exist[id] = 1;
  }

  return std::accumulate(exist.begin(), exist.end(), 0,
                         [](uint32_t a, uint32_t b) { return a + b; }) ==
         static_cast<int32_t>(map_size);
}

template <typename T>
auto is_valid_vector_optional(const std::vector<std::optional<T>> &optionals) {
  auto pred = [](const auto &elem) { return elem.has_value(); };
  if constexpr (std::is_same_v<T, TensorInfo>) {
    pred = [](const T &elem) {
      return elem.has_value() && (elem->second.producer_node.has_value() ||
                                  elem->second.has_explicit_added);
    };
  }

  return std::ranges::all_of(optionals, pred);
}
} // namespace

auto is_valid(const Graph &g) -> bool {
  if (g.input_names.empty() || g.output_names.empty() ||
      !is_valid_map(g.tensor_name_to_idx) ||
      !is_valid_map(g.node_name_to_idx) ||
      !is_valid_vector_optional(g.tensor_infos) ||
      !is_valid_vector_optional(g.nodes) ||
      std::ranges::any_of(g.input_names,
                          [&g](const auto &name) {
                            return !g.tensor_name_to_idx.contains(name) ||
                                   g.tensor_infos[g.tensor_name_to_idx.at(name)]
                                       ->second.producer_node.has_value() ||
                                   g.tensor_infos[g.tensor_name_to_idx.at(name)]
                                       ->second.consumer_nodes.empty();
                          }) ||
      std::ranges::any_of(
          g.output_names,
          [&g](const auto &name) {
            return !g.tensor_name_to_idx.contains(name) ||
                   !g.tensor_infos[g.tensor_name_to_idx.at(name)]
                        ->second.consumer_nodes.empty() ||
                   !g.tensor_infos[g.tensor_name_to_idx.at(name)]
                        ->second.producer_node;
          }) ||
      std::ranges::any_of(
          g.tensor_infos,
          [&g](const auto &tensor_info) {
            auto size = g.nodes.size();
            auto &producer_node = tensor_info->second.producer_node;
            return !g.tensor_name_to_idx.contains(tensor_info->first) ||
                   std::ranges::any_of(
                       tensor_info->second.consumer_nodes,
                       [size](auto id) { return id >= size; }) ||
                   (producer_node && *producer_node >= size) ||
                   (!producer_node && !tensor_info->second.has_explicit_added);
          }) ||
      std::ranges::any_of(g.nodes, [&g](const auto &node) {
        auto size = g.tensor_infos.size();
        auto pred = [size](auto id) { return id >= size; };
        return !g.node_name_to_idx.contains(node->first) ||
               std::ranges::any_of(node->second.input_tensors, pred) ||
               std::ranges::any_of(node->second.output_tensors, pred);
      })) {
    return false;
  }

  // Check if each output is reachable from the input.
  std::vector<bool> is_visited(g.tensor_infos.size(), false);
  std::queue<uint32_t> q;
  for (const auto &name : g.input_names) {
    q.push(g.tensor_name_to_idx.at(name));
  }
  while (!q.empty()) {
    auto cur_id = q.front();
    q.pop();

    if (is_visited[cur_id]) {
      continue;
    }

    is_visited[cur_id] = true;

    for (auto node_id : g.tensor_infos.at(cur_id)->second.consumer_nodes) {
      for (auto tensor_id : g.nodes.at(node_id)->second.output_tensors) {
        q.push(tensor_id);
      }
    }
  }

  return std::ranges::all_of(
      g.output_names, [&is_visited, &g](const auto &name) {
        return is_visited.at(g.tensor_name_to_idx.at(name));
      });
}
} // namespace tiny_llm
