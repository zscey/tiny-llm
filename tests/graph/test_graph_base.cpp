#include "tiny_llm/graph/graph.hpp"
#include "gtest/gtest.h"

namespace tiny_llm {
TEST(Graph, GraphBaseApi) {
  Graph graph;

  graph.add_tensor("input1", DataType::kFloat32, {});
  graph.add_node("node1",
                 {.input_names = {"input1", "input2", "input3"},
                  .output_names = {"node1_out1", "node1_out2"}},
                 AddParam{});
  graph.add_tensor("input2", DataType::kFloat32, {});

  EXPECT_ANY_THROW(graph.add_tensor("input2", DataType::kFloat32, {}));
  EXPECT_ANY_THROW(graph.add_node("node1", {}, AddParam{}));
  EXPECT_ANY_THROW((graph.set_input_names({"input1", "input4"})));
  EXPECT_ANY_THROW((graph.set_output_names({"output1", "output2"})));

  graph.set_input_names({"input1"});
  graph.set_output_names({"node1_out1"});

  EXPECT_EQ(graph.node_name_to_idx.size(), 1);
  EXPECT_EQ(graph.node_name_to_idx.at("node1"), 0);

  EXPECT_EQ(graph.tensor_name_to_idx.size(), 5);
  EXPECT_EQ(graph.tensor_name_to_idx.at("input1"), 0);
  EXPECT_EQ(graph.tensor_name_to_idx.at("input2"), 1);
  EXPECT_EQ(graph.tensor_name_to_idx.at("input3"), 2);
  EXPECT_EQ(graph.tensor_name_to_idx.at("node1_out1"), 3);
  EXPECT_EQ(graph.tensor_name_to_idx.at("node1_out2"), 4);

  EXPECT_EQ(graph.input_names, std::vector<std::string>{"input1"});

  EXPECT_EQ(graph.output_names, std::vector<std::string>{"node1_out1"});

  EXPECT_EQ(graph.nodes.size(), 1);
  const auto &node1 = graph.nodes.at(0).value();
  EXPECT_EQ(node1.input_tensors, (std::vector<uint32_t>{0, 1, 2}));
  EXPECT_EQ(node1.output_tensors, (std::vector<uint32_t>{3, 4}));
  EXPECT_EQ(node1.param.index(), 0);

  EXPECT_EQ(graph.tensor_infos.size(), 5);
  for (size_t i = 0; i < 2; ++i) {
    const auto &tensor_info = graph.tensor_infos[i].value();
    EXPECT_EQ(tensor_info.is_initialized, true);
    EXPECT_EQ(tensor_info.dtype, DataType::kFloat32);
    EXPECT_EQ(tensor_info.shape, std::vector<int64_t>{});
    EXPECT_FALSE(tensor_info.producer_node);
    EXPECT_EQ(tensor_info.consumer_nodes, std::vector<uint32_t>{0});
  }
  EXPECT_FALSE(graph.tensor_infos[2].value().is_initialized);
  EXPECT_FALSE(graph.tensor_infos[2].value().producer_node);
  EXPECT_EQ(graph.tensor_infos[2].value().consumer_nodes,
            std::vector<uint32_t>{0});
  for (size_t i = 3; i < 5; ++i) {
    const auto &tensor_info = graph.tensor_infos[i].value();
    EXPECT_EQ(tensor_info.is_initialized, false);
    EXPECT_EQ(tensor_info.producer_node.value(), 0);
    EXPECT_TRUE(tensor_info.consumer_nodes.empty());
  }
}
} // namespace tiny_llm
