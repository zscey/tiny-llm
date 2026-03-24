#include "tiny_llm/graph/graph_optimizer.hpp"
#include "graph.hpp"
#include "tiny_llm/common/log_and_excepts.hpp"
#include <algorithm>
#include <map>
#include <numeric>
#include <queue>
#include <set>

namespace tiny_llm {
namespace {
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void shrink(Graph &g, std::vector<uint32_t> retain_nodes) {
  // Shrink g.nodes
  std::ranges::sort(retain_nodes);
  TINY_LLM_CHECK(retain_nodes.back() < g.nodes.size());
  std::map<uint32_t, uint32_t> node_origin_to_new;
  for (uint32_t i = 0, i_end = retain_nodes.size(); i < i_end; ++i) {
    node_origin_to_new.emplace(retain_nodes[i], i);
  }
  for (auto [ori_id, new_id] : node_origin_to_new) {
    if (new_id != ori_id) {
      g.nodes[new_id] = std::move(g.nodes[ori_id]);
    }
  }
  g.nodes.resize(retain_nodes.size());

  // Shrink g.tensor_infos
  std::set<uint32_t> retain_tensor_infos;
  for (auto [_, new_node_id] : node_origin_to_new) {
    for (auto tensor_id : g.nodes.at(new_node_id)->second.input_tensors) {
      retain_tensor_infos.emplace(tensor_id);
    }
    for (auto tensor_id : g.nodes.at(new_node_id)->second.output_tensors) {
      retain_tensor_infos.emplace(tensor_id);
    }
  }
  std::map<uint32_t, uint32_t> tensor_origin_to_new;
  uint32_t new_id{};
  for (auto origin_id : retain_tensor_infos) {
    tensor_origin_to_new.emplace(origin_id, new_id++);
  }
  for (auto [ori_id, new_id] : tensor_origin_to_new) {
    if (new_id != ori_id) {
      g.tensor_infos[new_id] = std::move(g.tensor_infos[ori_id]);
    }
  }
  g.tensor_infos.resize(tensor_origin_to_new.size());

  // Shrink g.node_name_to_idx, modify g.node
  g.node_name_to_idx.clear();
  for (uint32_t node_id = 0, node_id_end = g.nodes.size();
       node_id < node_id_end; ++node_id) {
    auto &node = g.nodes.at(node_id);
    g.node_name_to_idx.emplace(node->first, node_id);
    for (auto &tensor_id : node->second.input_tensors) {
      tensor_id = tensor_origin_to_new.at(tensor_id);
    }
    for (auto &tensor_id : node->second.output_tensors) {
      tensor_id = tensor_origin_to_new.at(tensor_id);
    }
  }

  // Shrink g.tensor_name_to_idx, modify g.tensor_infos
  g.tensor_name_to_idx.clear();
  for (uint32_t tensor_id = 0, tensor_id_end = g.tensor_infos.size();
       tensor_id < tensor_id_end; ++tensor_id) {
    g.tensor_name_to_idx.emplace(g.tensor_infos.at(tensor_id)->first,
                                 tensor_id);
    auto &producer_node = g.tensor_infos.at(tensor_id)->second.producer_node;
    if (producer_node) {
      producer_node = node_origin_to_new.at(*producer_node);
    }

    uint32_t last_idx{};
    auto &consumer_nodes = g.tensor_infos.at(tensor_id)->second.consumer_nodes;
    for (uint32_t idx = 0, idx_end = consumer_nodes.size(); idx < idx_end;
         ++idx) {
      if (node_origin_to_new.contains(consumer_nodes[idx])) {
        consumer_nodes[last_idx++] = node_origin_to_new.at(consumer_nodes[idx]);
      }
    }
    consumer_nodes.resize(last_idx);
  }

  // Shrink g.input_names
  for (auto iter = g.input_names.begin(); iter != g.input_names.end();) {
    if (g.tensor_name_to_idx.contains(*iter)) {
      ++iter;
    } else {
      iter = g.input_names.erase(iter);
    }
  }
}
} // namespace

void PruningPass::run(Graph &g, WeightManagerWrapper & /*w*/) {
  (void)(this);

  std::vector<bool> is_visited(g.nodes.size(), false);

  std::queue<uint32_t> q;
  for (const auto &name : g.output_names) {
    q.push(g.tensor_infos.at(g.tensor_name_to_idx.at(name))
               ->second.producer_node.value());
  }
  while (!q.empty()) {
    auto cur_id = q.front();
    q.pop();

    if (is_visited[cur_id]) {
      continue;
    }

    is_visited[cur_id] = true;

    for (auto tensor_id : g.nodes.at(cur_id)->second.input_tensors) {
      auto &producer_node = g.tensor_infos.at(tensor_id)->second.producer_node;
      if (producer_node) {
        q.push(*producer_node);
      }
    }
  }

  std::vector<uint32_t> retain_nodes;
  for (uint32_t i = 0, i_end = is_visited.size(); i < i_end; ++i) {
    if (is_visited.at(i)) {
      retain_nodes.emplace_back(i);
    }
  }
  shrink(g, retain_nodes);
}

static_assert(GraphPass<PruningPass>);
} // namespace tiny_llm
