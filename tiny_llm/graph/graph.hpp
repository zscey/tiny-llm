#pragma once

#include "tiny_llm/graph/param.hpp"
#include "tiny_llm/tensor/tensor.hpp"
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tiny_llm {

struct NodeIONames {
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
};

class Graph {
public:
  class TensorInfo {
  public:
    bool has_explicit_added{false};
    DataType dtype;
    std::vector<int64_t> shape;
    std::vector<uint32_t> consumer_nodes;
    std::optional<uint32_t> producer_node;
  };

  class Node {
  public:
    std::vector<uint32_t> input_tensors;
    std::vector<uint32_t> output_tensors;

    Param param;
  };

  std::vector<std::optional<std::pair<std::string, Node>>> nodes;
  std::vector<std::optional<std::pair<std::string, TensorInfo>>> tensor_infos;

  std::unordered_map<std::string, uint32_t> node_name_to_idx;
  std::unordered_map<std::string, uint32_t> tensor_name_to_idx;
  std::unordered_set<std::string> input_names;
  std::unordered_set<std::string> output_names;

  void add_tensor(const std::string &name, DataType dtype,
                  std::vector<int64_t> shape);
  void add_node(const std::string &name, const NodeIONames &node_io_names,
                Param param);

  void set_input_names(std::unordered_set<std::string> input_names);
  void set_output_names(std::unordered_set<std::string> output_names);
};

auto is_valid(const Graph &g) -> bool;
} // namespace tiny_llm
