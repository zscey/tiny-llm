#include "tiny_llm/runtime/cuda/cuda_plan.hpp"
#include "gtest/gtest.h"
#include <cstdint>

namespace tiny_llm {
TEST(Runtime, CudaPlan) {
  Graph graph;
  graph.add_tensor("input1", DataType::kFloat32, {2, 3, 4, 5});
  graph.add_tensor("input2", DataType::kFloat32, {2, 3, 4, 5});
  graph.add_node(
      "node1",
      {.input_names = {"input1", "input2"}, .output_names = {"node1_output"}},
      AddParam{});
  graph.add_tensor("input3", DataType::kFloat32, {2, 3, 4, 5});
  graph.add_node("node2",
                 {.input_names = {"node1_output", "input3"},
                  .output_names = {"node2_output"}},
                 AddParam{});

  graph.set_input_names({"input2"});
  graph.set_output_names({"node2_output"});
  EXPECT_TRUE(is_valid(graph));

  auto cuda_plan = cuda::create_cuda_plan(
      graph, {.named_max_shapes = {{"input2", {2, 3, 4, 5}}}});

  { // Check input infos
    EXPECT_EQ(cuda_plan.input_infos.size(), 1);
    const auto &plan_input_0 = cuda_plan.input_infos.at("input2");
    EXPECT_EQ(std::get<0>(plan_input_0), 1);
    const auto &plan_input_0_task_infos = std::get<1>(plan_input_0);
    EXPECT_EQ(plan_input_0_task_infos.size(), 1);
    EXPECT_EQ(plan_input_0_task_infos.at(0).task_id, 0);
    EXPECT_EQ(plan_input_0_task_infos.at(0).io_id, 1);
  }

  { // Check output infos
    EXPECT_EQ(cuda_plan.output_infos.size(), 1);
    EXPECT_EQ(std::get<0>(cuda_plan.output_infos.at("node2_output")), 4);
    EXPECT_EQ(std::get<1>(cuda_plan.output_infos.at("node2_output")).task_id,
              1);
    EXPECT_EQ(std::get<1>(cuda_plan.output_infos.at("node2_output")).io_id, 0);
  }

  {
    // Check descs
    EXPECT_EQ(cuda_plan.tensor_descs.size(), 5);
    for (const auto &desc : cuda_plan.tensor_descs) {
      EXPECT_EQ(desc.dtype, DataType::kFloat32);
      EXPECT_EQ(desc.cur_shape, (std::vector<size_t>{2, 3, 4, 5}));
      EXPECT_EQ(desc.max_shape, (std::vector<size_t>{2, 3, 4, 5}));
    }
  }

  { // Check tasks
    EXPECT_EQ(cuda_plan.tasks.size(), 2);
    const auto &task1 = cuda_plan.tasks[0];
    EXPECT_EQ(task1.name, "node1");
    EXPECT_EQ(task1.kernel.index(), 0);
    EXPECT_TRUE(task1.predecessors.empty());
    EXPECT_EQ(task1.successors, std::vector<uint32_t>{1});
    EXPECT_EQ(task1.input_descs,
              (std::vector<const TensorDesc *>{&cuda_plan.tensor_descs.at(0),
                                               &cuda_plan.tensor_descs.at(1)}));
    EXPECT_EQ(task1.output_descs,
              (std::vector<TensorDesc *>{&cuda_plan.tensor_descs.at(2)}));
    EXPECT_EQ(cuda_plan.tasks.size(), 2);
    const auto &task2 = cuda_plan.tasks[1];
    EXPECT_EQ(task2.name, "node2");
    EXPECT_EQ(task2.kernel.index(), 0);
    EXPECT_EQ(task2.predecessors, std::vector<uint32_t>{0});
    EXPECT_TRUE(task2.successors.empty());
    EXPECT_EQ(task2.input_descs,
              (std::vector<const TensorDesc *>{&cuda_plan.tensor_descs.at(2),
                                               &cuda_plan.tensor_descs.at(3)}));
    EXPECT_EQ(task2.output_descs,
              (std::vector<TensorDesc *>{&cuda_plan.tensor_descs.at(4)}));
  }
}

} // namespace tiny_llm
