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
      {.input_names = {"input1", "input2"}, .output_names = {"output"}},
      AddParam{});
  graph.set_input_names({"input2"});
  graph.set_output_names({"output"});
  EXPECT_TRUE(is_valid(graph));

  auto cuda_plan = cuda::create_cuda_plan(
      graph, {.named_max_shapes = {{"input2", {2, 3, 4, 5}}}});
  EXPECT_EQ(cuda_plan.input_infos.size(), 1);
  EXPECT_EQ(
      cuda_plan.input_infos.at("input2"),
      (std::make_tuple<uint32_t, std::vector<std::pair<uint32_t, uint32_t>>>(
          1, std::vector<std::pair<uint32_t, uint32_t>>{{0, 1}})));
  EXPECT_EQ(cuda_plan.output_infos.size(), 1);
  EXPECT_EQ(cuda_plan.output_infos.at("output"), 2);

  EXPECT_EQ(cuda_plan.tensor_descs.size(), 3);
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(cuda_plan.tensor_descs.at(i).dtype, DataType::kFloat32);
    EXPECT_EQ(cuda_plan.tensor_descs.at(i).cur_shape,
              (std::vector<size_t>{2, 3, 4, 5}));
    EXPECT_EQ(cuda_plan.tensor_descs.at(i).max_shape,
              (std::vector<size_t>{2, 3, 4, 5}));
  }

  EXPECT_EQ(cuda_plan.tasks.size(), 1);
  const auto &task = cuda_plan.tasks.back();
  EXPECT_EQ(task.kernel.index(), 0);
  EXPECT_TRUE(task.predecessors.empty());
  EXPECT_TRUE(task.successors.empty());
  EXPECT_EQ(task.input_descs,
            (std::vector<const TensorDesc *>{&cuda_plan.tensor_descs.at(0),
                                             &cuda_plan.tensor_descs.at(1)}));
  EXPECT_EQ(task.output_descs,
            (std::vector<TensorDesc *>{&cuda_plan.tensor_descs.at(2)}));
}

} // namespace tiny_llm
