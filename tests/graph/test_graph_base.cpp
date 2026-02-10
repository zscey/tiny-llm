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
                 {.input_names = {"input1"}, .output_names = {"node1_out"}},
                 SiLUParam{});
  graph.add_node("node2",
                 {.input_names = {"input1"}, .output_names = {"node2_out"}},
                 SiLUParam{});
  graph.add_node("node3",
                 {.input_names = {"node2_out"}, .output_names = {"node3_out"}},
                 SiLUParam{});
  graph.add_node("node4",
                 {.input_names = {"input2"}, .output_names = {"node4_out"}},
                 SiLUParam{});
  {
    EXPECT_ANY_THROW(graph.add_tensor("input1", DataType::kFloat32, {}));
    EXPECT_ANY_THROW(graph.add_node(
        "node1", {.input_names = {"input1"}, .output_names = {"node1_out"}},
        SiLUParam{}));
    EXPECT_ANY_THROW((graph.set_input_names({"input1", "input4"})));
    EXPECT_ANY_THROW((graph.set_output_names({"output1", "output2"})));
  }
  graph.set_input_names({"input1", "input2"});
  graph.set_output_names({"node2_out"});
  EXPECT_FALSE(is_valid(graph));
  graph.add_tensor("input2", DataType::kFloat32, {});
  // invalid graph: node2_out -> node3
  EXPECT_FALSE(is_valid(graph));

  {
    EXPECT_EQ(graph.node_name_to_idx.size(), 4);
    EXPECT_EQ(graph.node_name_to_idx.at("node1"), 0);
    EXPECT_EQ(graph.node_name_to_idx.at("node2"), 1);
    EXPECT_EQ(graph.node_name_to_idx.at("node3"), 2);
    EXPECT_EQ(graph.node_name_to_idx.at("node4"), 3);
    EXPECT_EQ(graph.tensor_name_to_idx.size(), 6);
    EXPECT_EQ(graph.tensor_name_to_idx.at("input1"), 0);
    EXPECT_EQ(graph.tensor_name_to_idx.at("node1_out"), 1);
    EXPECT_EQ(graph.tensor_name_to_idx.at("node2_out"), 2);
    EXPECT_EQ(graph.tensor_name_to_idx.at("node3_out"), 3);
    EXPECT_EQ(graph.tensor_name_to_idx.at("input2"), 4);
    EXPECT_EQ(graph.tensor_name_to_idx.at("node4_out"), 5);
    EXPECT_EQ(graph.input_names,
              (std::unordered_set<std::string>{"input1", "input2"}));
    EXPECT_EQ(graph.output_names, std::unordered_set<std::string>{"node2_out"});

    EXPECT_EQ(graph.nodes.size(), 4);
    const auto &node0 = graph.nodes.at(0).value();
    EXPECT_EQ(node0.first, "node1");
    EXPECT_EQ(node0.second.input_tensors, (std::vector<uint32_t>{0}));
    EXPECT_EQ(node0.second.output_tensors, (std::vector<uint32_t>{1}));
    EXPECT_EQ(node0.second.param.index(), 0);
    const auto &node1 = graph.nodes.at(1).value();
    EXPECT_EQ(node1.first, "node2");
    EXPECT_EQ(node1.second.input_tensors, (std::vector<uint32_t>{0}));
    EXPECT_EQ(node1.second.output_tensors, (std::vector<uint32_t>{2}));
    EXPECT_EQ(node1.second.param.index(), 0);
    const auto &node2 = graph.nodes.at(2).value();
    EXPECT_EQ(node2.first, "node3");
    EXPECT_EQ(node2.second.input_tensors, (std::vector<uint32_t>{2}));
    EXPECT_EQ(node2.second.output_tensors, (std::vector<uint32_t>{3}));
    EXPECT_EQ(node2.second.param.index(), 0);
    const auto &node3 = graph.nodes.at(3).value();
    EXPECT_EQ(node3.first, "node4");
    EXPECT_EQ(node3.second.input_tensors, (std::vector<uint32_t>{4}));
    EXPECT_EQ(node3.second.output_tensors, (std::vector<uint32_t>{5}));
    EXPECT_EQ(node3.second.param.index(), 0);

    EXPECT_EQ(graph.tensor_infos.size(), 6);
    const auto &tensor_info0 = graph.tensor_infos.at(0).value();
    EXPECT_EQ(tensor_info0.first, "input1");
    EXPECT_EQ(tensor_info0.second.has_explicit_added, true);
    EXPECT_EQ(tensor_info0.second.dtype, DataType::kFloat32);
    EXPECT_EQ(tensor_info0.second.shape, std::vector<int64_t>{});
    EXPECT_FALSE(tensor_info0.second.producer_node);
    EXPECT_EQ(tensor_info0.second.consumer_nodes,
              (std::vector<uint32_t>{0, 1}));
    const auto &tensor_info1 = graph.tensor_infos.at(1).value();
    EXPECT_EQ(tensor_info1.first, "node1_out");
    EXPECT_EQ(tensor_info1.second.has_explicit_added, false);
    EXPECT_EQ(*tensor_info1.second.producer_node, 0);
    EXPECT_TRUE(tensor_info1.second.consumer_nodes.empty());
    const auto &tensor_info2 = graph.tensor_infos.at(2).value();
    EXPECT_EQ(tensor_info2.first, "node2_out");
    EXPECT_EQ(tensor_info2.second.has_explicit_added, false);
    EXPECT_EQ(*tensor_info2.second.producer_node, 1);
    EXPECT_EQ(tensor_info2.second.consumer_nodes, std::vector<uint32_t>{2});
    const auto &tensor_info3 = graph.tensor_infos.at(3).value();
    EXPECT_EQ(tensor_info3.first, "node3_out");
    EXPECT_EQ(tensor_info3.second.has_explicit_added, false);
    EXPECT_EQ(*tensor_info3.second.producer_node, 2);
    EXPECT_TRUE(tensor_info3.second.consumer_nodes.empty());
    const auto &tensor_info4 = graph.tensor_infos.at(4).value();
    EXPECT_EQ(tensor_info4.first, "input2");
    EXPECT_EQ(tensor_info4.second.has_explicit_added, true);
    EXPECT_EQ(tensor_info4.second.dtype, DataType::kFloat32);
    EXPECT_EQ(tensor_info4.second.shape, std::vector<int64_t>{});
    EXPECT_FALSE(tensor_info4.second.producer_node);
    EXPECT_EQ(tensor_info4.second.consumer_nodes, std::vector<uint32_t>{3});
    const auto &tensor_info5 = graph.tensor_infos.at(5).value();
    EXPECT_EQ(tensor_info5.first, "node4_out");
    EXPECT_EQ(tensor_info5.second.has_explicit_added, false);
    EXPECT_EQ(*tensor_info5.second.producer_node, 3);
    EXPECT_TRUE(tensor_info5.second.consumer_nodes.empty());
  }

  PassManager pass_manager;
  pass_manager.add_pass(PruningPass{});
  WeightManagerWrapper weight_manager(
      (SafeTensorWeightManager(utils::BazelRunfile::RLocation(
          "tiny_llm/tests/datas/test.safetensors"))));
  pass_manager.run(graph, weight_manager);
  EXPECT_TRUE(is_valid(graph));
  {
    EXPECT_EQ(graph.node_name_to_idx.size(), 1);
    EXPECT_EQ(graph.node_name_to_idx.at("node2"), 0);
    EXPECT_EQ(graph.tensor_name_to_idx.size(), 2);
    EXPECT_EQ(graph.tensor_name_to_idx.at("input1"), 0);
    EXPECT_EQ(graph.tensor_name_to_idx.at("node2_out"), 1);
    EXPECT_EQ(graph.input_names, (std::unordered_set<std::string>{"input1"}));
    EXPECT_EQ(graph.output_names, std::unordered_set<std::string>{"node2_out"});

    EXPECT_EQ(graph.nodes.size(), 1);
    const auto &node0 = graph.nodes.at(0).value();
    EXPECT_EQ(node0.first, "node2");
    EXPECT_EQ(node0.second.input_tensors, (std::vector<uint32_t>{0}));
    EXPECT_EQ(node0.second.output_tensors, (std::vector<uint32_t>{1}));
    EXPECT_EQ(node0.second.param.index(), 0);

    EXPECT_EQ(graph.tensor_infos.size(), 2);
    const auto &tensor_info0 = graph.tensor_infos.at(0).value();
    EXPECT_EQ(tensor_info0.first, "input1");
    EXPECT_EQ(tensor_info0.second.has_explicit_added, true);
    EXPECT_EQ(tensor_info0.second.dtype, DataType::kFloat32);
    EXPECT_EQ(tensor_info0.second.shape, std::vector<int64_t>{});
    EXPECT_FALSE(tensor_info0.second.producer_node);
    EXPECT_EQ(tensor_info0.second.consumer_nodes, std::vector<uint32_t>{0});
    const auto &tensor_info1 = graph.tensor_infos.at(1).value();
    EXPECT_EQ(tensor_info1.first, "node2_out");
    EXPECT_EQ(tensor_info1.second.has_explicit_added, false);
    EXPECT_EQ(*tensor_info1.second.producer_node, 0);
    EXPECT_TRUE(tensor_info1.second.consumer_nodes.empty());
  }
}
} // namespace tiny_llm
