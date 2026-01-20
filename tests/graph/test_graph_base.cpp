#include "tiny_llm/graph/graph.hpp"
#include "tiny_llm/graph/graph_optimizer.hpp"
#include "tiny_llm/utils/runfile.hpp"
#include "tiny_llm/weight_managers/safetensors/weight_manager.hpp"
#include "gtest/gtest.h"

namespace tiny_llm {
TEST(Graph, GraphBaseApi) {
  // Construct graph
  Graph graph;
  graph.add_tensor("input1", DataType::kFloat32, {});
  graph.add_node("node1",
                 {.input_names = {"input1", "input2"},
                  .output_names = {"node1_out1", "node1_out2"}},
                 AddParam{});
  graph.add_node("node2",
                 {.input_names = {"input3"}, .output_names = {"node2_out1"}},
                 AddParam{});
  graph.add_node(
      "node3", {.input_names = {"node1_out2"}, .output_names = {"node3_out1"}},
      AddParam{});
  graph.add_tensor("input2", DataType::kFloat32, {});
  graph.add_node(
      "node4", {.input_names = {"node3_out1"}, .output_names = {"node4_out1"}},
      AddParam{});
  {
    EXPECT_ANY_THROW(graph.add_tensor("input2", DataType::kFloat32, {}));
    EXPECT_ANY_THROW(graph.add_node("node1", {}, AddParam{}));
    EXPECT_ANY_THROW((graph.set_input_names({"input1", "input4"})));
    EXPECT_ANY_THROW((graph.set_output_names({"output1", "output2"})));
  }
  graph.set_input_names({"input1", "input2", "input3"});
  graph.set_output_names({"node3_out1"});
  EXPECT_FALSE(is_valid(graph));
  graph.add_tensor("input3", DataType::kFloat32, {});
  // invalid graph: node3_out1 -> node4
  EXPECT_FALSE(is_valid(graph));

  {
    EXPECT_EQ(graph.node_name_to_idx.size(), 4);
    EXPECT_EQ(graph.node_name_to_idx.at("node1"), 0);
    EXPECT_EQ(graph.node_name_to_idx.at("node2"), 1);
    EXPECT_EQ(graph.node_name_to_idx.at("node3"), 2);
    EXPECT_EQ(graph.tensor_name_to_idx.size(), 8);
    EXPECT_EQ(graph.tensor_name_to_idx.at("input1"), 0);
    EXPECT_EQ(graph.tensor_name_to_idx.at("input2"), 1);
    EXPECT_EQ(graph.tensor_name_to_idx.at("node1_out1"), 2);
    EXPECT_EQ(graph.tensor_name_to_idx.at("node1_out2"), 3);
    EXPECT_EQ(graph.tensor_name_to_idx.at("input3"), 4);
    EXPECT_EQ(graph.tensor_name_to_idx.at("node2_out1"), 5);
    EXPECT_EQ(graph.tensor_name_to_idx.at("node3_out1"), 6);
    EXPECT_EQ(graph.tensor_name_to_idx.at("node4_out1"), 7);
    EXPECT_EQ(graph.input_names,
              (std::unordered_set<std::string>{"input1", "input2", "input3"}));
    EXPECT_EQ(graph.output_names,
              std::unordered_set<std::string>{"node3_out1"});
    EXPECT_EQ(graph.nodes.size(), 4);
    const auto &node0 = graph.nodes.at(0).value();
    EXPECT_EQ(node0.first, "node1");
    EXPECT_EQ(node0.second.input_tensors, (std::vector<uint32_t>{0, 1}));
    EXPECT_EQ(node0.second.output_tensors, (std::vector<uint32_t>{2, 3}));
    EXPECT_EQ(node0.second.param.index(), 0);
    const auto &node1 = graph.nodes.at(1).value();
    EXPECT_EQ(node1.first, "node2");
    EXPECT_EQ(node1.second.input_tensors, (std::vector<uint32_t>{4}));
    EXPECT_EQ(node1.second.output_tensors, (std::vector<uint32_t>{5}));
    EXPECT_EQ(node1.second.param.index(), 0);
    const auto &node2 = graph.nodes.at(2).value();
    EXPECT_EQ(node2.first, "node3");
    EXPECT_EQ(node2.second.input_tensors, (std::vector<uint32_t>{3}));
    EXPECT_EQ(node2.second.output_tensors, (std::vector<uint32_t>{6}));
    EXPECT_EQ(node2.second.param.index(), 0);
    const auto &node3 = graph.nodes.at(3).value();
    EXPECT_EQ(node3.first, "node4");
    EXPECT_EQ(node3.second.input_tensors, (std::vector<uint32_t>{6}));
    EXPECT_EQ(node3.second.output_tensors, (std::vector<uint32_t>{7}));
    EXPECT_EQ(node3.second.param.index(), 0);

    EXPECT_EQ(graph.tensor_infos.size(), 8);
    const auto &tensor_info0 = graph.tensor_infos.at(0).value();
    EXPECT_EQ(tensor_info0.first, "input1");
    EXPECT_EQ(tensor_info0.second.has_explicit_added, true);
    EXPECT_EQ(tensor_info0.second.dtype, DataType::kFloat32);
    EXPECT_EQ(tensor_info0.second.shape, std::vector<int64_t>{});
    EXPECT_FALSE(tensor_info0.second.producer_node);
    EXPECT_EQ(tensor_info0.second.consumer_nodes, std::vector<uint32_t>{0});
    const auto &tensor_info1 = graph.tensor_infos.at(1).value();
    EXPECT_EQ(tensor_info1.first, "input2");
    EXPECT_EQ(tensor_info1.second.has_explicit_added, true);
    EXPECT_EQ(tensor_info1.second.dtype, DataType::kFloat32);
    EXPECT_EQ(tensor_info1.second.shape, std::vector<int64_t>{});
    EXPECT_FALSE(tensor_info1.second.producer_node);
    EXPECT_EQ(tensor_info1.second.consumer_nodes, std::vector<uint32_t>{0});
    const auto &tensor_info2 = graph.tensor_infos.at(2).value();
    EXPECT_EQ(tensor_info2.first, "node1_out1");
    EXPECT_EQ(tensor_info2.second.has_explicit_added, false);
    EXPECT_EQ(*tensor_info2.second.producer_node, 0);
    EXPECT_TRUE(tensor_info2.second.consumer_nodes.empty());
    const auto &tensor_info3 = graph.tensor_infos.at(3).value();
    EXPECT_EQ(tensor_info3.first, "node1_out2");
    EXPECT_EQ(tensor_info3.second.has_explicit_added, false);
    EXPECT_EQ(*tensor_info3.second.producer_node, 0);
    EXPECT_EQ(tensor_info3.second.consumer_nodes, std::vector<uint32_t>{2});
    const auto &tensor_info4 = graph.tensor_infos.at(4).value();
    EXPECT_EQ(tensor_info4.first, "input3");
    EXPECT_EQ(tensor_info4.second.has_explicit_added, true);
    EXPECT_EQ(tensor_info4.second.dtype, DataType::kFloat32);
    EXPECT_EQ(tensor_info4.second.shape, std::vector<int64_t>{});
    EXPECT_FALSE(tensor_info4.second.producer_node);
    EXPECT_EQ(tensor_info4.second.consumer_nodes, std::vector<uint32_t>{1});
    const auto &tensor_info5 = graph.tensor_infos.at(5).value();
    EXPECT_EQ(tensor_info5.first, "node2_out1");
    EXPECT_EQ(tensor_info5.second.has_explicit_added, false);
    EXPECT_EQ(*tensor_info5.second.producer_node, 1);
    EXPECT_TRUE(tensor_info5.second.consumer_nodes.empty());
    const auto &tensor_info6 = graph.tensor_infos.at(6).value();
    EXPECT_EQ(tensor_info6.first, "node3_out1");
    EXPECT_EQ(tensor_info6.second.has_explicit_added, false);
    EXPECT_EQ(*tensor_info6.second.producer_node, 2);
    EXPECT_TRUE(tensor_info6.second.consumer_nodes == std::vector<uint32_t>{3});
    const auto &tensor_info7 = graph.tensor_infos.at(7).value();
    EXPECT_EQ(tensor_info7.first, "node4_out1");
    EXPECT_EQ(tensor_info7.second.has_explicit_added, false);
    EXPECT_EQ(*tensor_info7.second.producer_node, 3);
    EXPECT_TRUE(tensor_info7.second.consumer_nodes.empty());
  }

  PassManager pass_manager;
  pass_manager.add_pass(PruningPass{});
  WeightManagerWrapper weight_manager(
      (SafeTensorWeightManager(utils::BazelRunfile::RLocation(
          "tiny_llm/tests/datas/test.safetensors"))));
  pass_manager.run(graph, weight_manager);
  {
    EXPECT_EQ(graph.node_name_to_idx.size(), 2);
    EXPECT_EQ(graph.node_name_to_idx.at("node1"), 0);
    EXPECT_EQ(graph.node_name_to_idx.at("node3"), 1);
    EXPECT_EQ(graph.tensor_name_to_idx.size(), 5);
    EXPECT_EQ(graph.tensor_name_to_idx.at("input1"), 0);
    EXPECT_EQ(graph.tensor_name_to_idx.at("input2"), 1);
    EXPECT_EQ(graph.tensor_name_to_idx.at("node1_out1"), 2);
    EXPECT_EQ(graph.tensor_name_to_idx.at("node1_out2"), 3);
    EXPECT_EQ(graph.tensor_name_to_idx.at("node3_out1"), 4);
    EXPECT_EQ(graph.input_names,
              (std::unordered_set<std::string>{"input1", "input2"}));
    EXPECT_EQ(graph.output_names,
              std::unordered_set<std::string>{"node3_out1"});
    EXPECT_EQ(graph.nodes.size(), 2);
    const auto &node0 = graph.nodes.at(0).value();
    EXPECT_EQ(node0.first, "node1");
    EXPECT_EQ(node0.second.input_tensors, (std::vector<uint32_t>{0, 1}));
    EXPECT_EQ(node0.second.output_tensors, (std::vector<uint32_t>{2, 3}));
    EXPECT_EQ(node0.second.param.index(), 0);
    const auto &node1 = graph.nodes.at(1).value();
    EXPECT_EQ(node1.first, "node3");
    EXPECT_EQ(node1.second.input_tensors, (std::vector<uint32_t>{3}));
    EXPECT_EQ(node1.second.output_tensors, (std::vector<uint32_t>{4}));
    EXPECT_EQ(node1.second.param.index(), 0);

    EXPECT_EQ(graph.tensor_infos.size(), 5);
    const auto &tensor_info0 = graph.tensor_infos.at(0).value();
    EXPECT_EQ(tensor_info0.first, "input1");
    EXPECT_EQ(tensor_info0.second.has_explicit_added, true);
    EXPECT_EQ(tensor_info0.second.dtype, DataType::kFloat32);
    EXPECT_EQ(tensor_info0.second.shape, std::vector<int64_t>{});
    EXPECT_FALSE(tensor_info0.second.producer_node);
    EXPECT_EQ(tensor_info0.second.consumer_nodes, std::vector<uint32_t>{0});
    const auto &tensor_info1 = graph.tensor_infos.at(1).value();
    EXPECT_EQ(tensor_info1.first, "input2");
    EXPECT_EQ(tensor_info1.second.has_explicit_added, true);
    EXPECT_EQ(tensor_info1.second.dtype, DataType::kFloat32);
    EXPECT_EQ(tensor_info1.second.shape, std::vector<int64_t>{});
    EXPECT_FALSE(tensor_info1.second.producer_node);
    EXPECT_EQ(tensor_info1.second.consumer_nodes, std::vector<uint32_t>{0});
    const auto &tensor_info2 = graph.tensor_infos.at(2).value();
    EXPECT_EQ(tensor_info2.first, "node1_out1");
    EXPECT_EQ(tensor_info2.second.has_explicit_added, false);
    EXPECT_EQ(*tensor_info2.second.producer_node, 0);
    EXPECT_TRUE(tensor_info2.second.consumer_nodes.empty());
    const auto &tensor_info3 = graph.tensor_infos.at(3).value();
    EXPECT_EQ(tensor_info3.first, "node1_out2");
    EXPECT_EQ(tensor_info3.second.has_explicit_added, false);
    EXPECT_EQ(*tensor_info3.second.producer_node, 0);
    EXPECT_EQ(tensor_info3.second.consumer_nodes, std::vector<uint32_t>{1});
    const auto &tensor_info4 = graph.tensor_infos.at(4).value();
    EXPECT_EQ(tensor_info4.first, "node3_out1");
    EXPECT_EQ(tensor_info4.second.has_explicit_added, false);
    EXPECT_EQ(*tensor_info4.second.producer_node, 1);
    EXPECT_TRUE(tensor_info4.second.consumer_nodes.empty());
  }
}
} // namespace tiny_llm
