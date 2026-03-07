#include "tiny_llm/graph/graph.hpp"
#include "tiny_llm/common/log_and_excepts.hpp"
#include <algorithm>
#include <numeric>
#include <queue>

namespace tiny_llm {

namespace {
auto get_tensor(Graph &graph, const std::string &name) -> Graph::TensorInfo & {
  if (auto iter = graph.tensor_name_to_idx.find(name);
      iter != graph.tensor_name_to_idx.end()) {
    auto &named_tensor_info = graph.tensor_infos[iter->second];
    TINY_LLM_CHECK(named_tensor_info);
    return named_tensor_info->second;
  }

  graph.tensor_name_to_idx.emplace(name, graph.tensor_infos.size());
  auto &named_tensor_info =
      graph.tensor_infos.emplace_back(std::in_place, name, Graph::TensorInfo{});
  return named_tensor_info->second;
}
} // namespace

void Graph::add_tensor(const std::string &name, DataType dtype,
                       std::vector<int64_t> shape) {
  auto &named_tensor_info = get_tensor(*this, name);
  TINY_LLM_CHECK(named_tensor_info.has_explicit_added == false);

  named_tensor_info.has_explicit_added = true;
  named_tensor_info.dtype = dtype;
  named_tensor_info.shape = std::move(shape);
}

void Graph::add_node(const std::string &name, const NodeIONames &node_io_names,
                     Param param) {
  TINY_LLM_CHECK(!node_name_to_idx.contains(name));
  TINY_LLM_CHECK(node_io_names.input_names.size() == input_num(param));
  TINY_LLM_CHECK(node_io_names.output_names.size() == output_num(param));

  auto node_id = nodes.size();
  node_name_to_idx.emplace(name, node_id);
  // TODO(): construct by move
  auto &named_node =
      nodes.emplace_back(std::in_place, name, Node{.param = param});

  for (const auto &name : node_io_names.input_names) {
    auto &named_tensor_info = get_tensor(*this, name);
    named_tensor_info.consumer_nodes.emplace_back(node_id);
    named_node->second.input_tensors.emplace_back(tensor_name_to_idx.at(name));
  }

  for (const auto &name : node_io_names.output_names) {
    auto &named_tensor_info = get_tensor(*this, name);
    TINY_LLM_CHECK(!named_tensor_info.producer_node);
    named_tensor_info.producer_node = node_id;
    named_node->second.output_tensors.emplace_back(tensor_name_to_idx.at(name));
  }
}

void Graph::set_input_names(std::unordered_set<std::string> input_names) {
  TINY_LLM_CHECK(
      std::ranges::all_of(input_names, [this](const auto &name) -> auto {
        return tensor_name_to_idx.contains(name);
      }))

  this->input_names = std::move(input_names);
}

void Graph::set_output_names(std::unordered_set<std::string> output_names) {
  TINY_LLM_CHECK(
      std::ranges::all_of(output_names, [this](const auto &name) -> auto {
        return tensor_name_to_idx.contains(name);
      }))

  this->output_names = std::move(output_names);
}

namespace {
auto is_valid_map(const std::unordered_map<std::string, uint32_t> &map)
    -> bool {
  auto map_size = map.size();

  std::vector<int32_t> exist(map_size, 0);
  for (const auto &[_, id] : map) {
    if (id >= map_size) {
      return false;
    }
    exist[id] = 1;
  }

  return std::accumulate(exist.begin(), exist.end(), 0,
                         [](auto a, auto b) -> auto { return a + b; }) ==
         static_cast<int32_t>(map_size);
}
} // namespace

auto is_valid(const Graph &g) -> bool {
  if (g.input_names.empty() || g.output_names.empty() ||
      !is_valid_map(g.tensor_name_to_idx) ||
      !is_valid_map(g.node_name_to_idx) ||
      g.tensor_name_to_idx.size() != g.tensor_infos.size() ||
      g.node_name_to_idx.size() != g.nodes.size() ||
      std::ranges::any_of(
          g.tensor_infos,
          [&g](const auto &named_tensor_info) -> auto {
            if (!named_tensor_info) {
              return true;
            }
            auto size = g.nodes.size();
            const auto &tensor_info = named_tensor_info->second;
            const auto &producer_node = tensor_info.producer_node;
            return !g.tensor_name_to_idx.contains(named_tensor_info->first) ||
                   std::ranges::any_of(
                       tensor_info.consumer_nodes,
                       [size](auto id) -> auto { return id >= size; }) ||
                   (producer_node && *producer_node >= size) ||
                   (!producer_node && !tensor_info.has_explicit_added);
          }) ||
      std::ranges::any_of(
          g.nodes,
          [&g](const auto &named_node) -> auto {
            if (!named_node) {
              return true;
            }
            auto size = g.tensor_infos.size();
            auto pred = [size](auto id) -> auto { return id >= size; };
            return !g.node_name_to_idx.contains(named_node->first) ||
                   std::ranges::any_of(named_node->second.input_tensors,
                                       pred) ||
                   std::ranges::any_of(named_node->second.output_tensors, pred);
          }) ||
      std::ranges::any_of(g.input_names,
                          [&g](const auto &name) -> auto {
                            const auto idx_iter =
                                g.tensor_name_to_idx.find(name);
                            if (idx_iter == g.tensor_name_to_idx.end()) {
                              return true;
                            };
                            const auto &tensor_info =
                                g.tensor_infos.at(idx_iter->second)->second;
                            return tensor_info.producer_node ||
                                   tensor_info.consumer_nodes.empty();
                          }) ||
      std::ranges::any_of(g.output_names, [&g](const auto &name) -> auto {
        const auto idx_iter = g.tensor_name_to_idx.find(name);
        if (idx_iter == g.tensor_name_to_idx.end()) {
          return true;
        }
        const auto &tensor_info = g.tensor_infos.at(idx_iter->second)->second;
        return !tensor_info.producer_node ||
               !tensor_info.consumer_nodes.empty();
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
      g.output_names, [&is_visited, &g](const auto &name) -> auto {
        return is_visited.at(g.tensor_name_to_idx.at(name));
      });
}
} // namespace tiny_llm
