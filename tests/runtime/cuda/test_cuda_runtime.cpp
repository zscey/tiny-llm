#include "tiny_llm/runtime/cuda/cuda_runtime.hpp"
#include "tiny_llm/utils/runfile.hpp"
#include "tiny_llm/weight_managers/safetensors/weight_manager.hpp"
#include "gtest/gtest.h"
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace tiny_llm {
namespace {
auto silu(float f) -> float { return f / (1 + std::exp(-f)); }
} // namespace

TEST(Runtime, CudaRuntime) {
  Graph graph;
  graph.add_tensor("input1", DataType::kFloat32, {2, 3, 4, 5});
  graph.add_node("node1",
                 {.input_names = {"input1"}, .output_names = {"node1_out"}},
                 SiLUParam{});
  graph.add_node("node2",
                 {.input_names = {"input1"}, .output_names = {"node2_out"}},
                 SiLUParam{});
  graph.add_node("node3",
                 {.input_names = {"node2_out"}, .output_names = {"node3_out"}},
                 SiLUParam{.inplace = true});

  graph.set_input_names({"input1"});
  graph.set_output_names({"node1_out", "node3_out"});
  EXPECT_TRUE(is_valid(graph));

  auto cuda_plan = cuda::create_cuda_plan(
      graph, {.named_max_shapes = {{"input1", {2, 3, 4, 8}}}});
  // topo res: node2->node3->node1
  { // Check input infos
    EXPECT_EQ(cuda_plan.input_infos.size(), 1);
    const auto &plan_input_0 = cuda_plan.input_infos.at("input1");
    EXPECT_EQ(std::get<0>(plan_input_0), 0);
    const auto &plan_input_0_task_infos = std::get<1>(plan_input_0);
    EXPECT_EQ(plan_input_0_task_infos.size(), 2);
    EXPECT_EQ(plan_input_0_task_infos.at(0).task_id, 0);
    EXPECT_EQ(plan_input_0_task_infos.at(0).io_id, 0);
    EXPECT_EQ(plan_input_0_task_infos.at(1).task_id, 2);
    EXPECT_EQ(plan_input_0_task_infos.at(1).io_id, 0);
  }

  { // Check output infos
    EXPECT_EQ(cuda_plan.output_infos.size(), 2);
    EXPECT_EQ(std::get<0>(cuda_plan.output_infos.at("node1_out")), 1);
    EXPECT_EQ(std::get<1>(cuda_plan.output_infos.at("node1_out")).task_id, 2);
    EXPECT_EQ(std::get<1>(cuda_plan.output_infos.at("node1_out")).io_id, 0);
    EXPECT_EQ(std::get<0>(cuda_plan.output_infos.at("node3_out")), 3);
    EXPECT_EQ(std::get<1>(cuda_plan.output_infos.at("node3_out")).task_id, 1);
    EXPECT_EQ(std::get<1>(cuda_plan.output_infos.at("node3_out")).io_id, 0);
  }

  {
    // Check descs
    EXPECT_EQ(cuda_plan.tensor_descs.size(), 4);
    for (const auto &desc : cuda_plan.tensor_descs) {
      EXPECT_EQ(desc.dtype, DataType::kFloat32);
      EXPECT_EQ(desc.cur_shape, (std::vector<size_t>{2, 3, 4, 8}));
      EXPECT_EQ(desc.max_shape, (std::vector<size_t>{2, 3, 4, 8}));
    }
  }

  {
    // Check dependence
    EXPECT_EQ(cuda_plan.tensor_dependence.size(), 4);
    EXPECT_EQ(cuda_plan.tensor_dependence.at(0), 2);
    EXPECT_EQ(cuda_plan.tensor_dependence.at(1), 0);
    EXPECT_EQ(cuda_plan.tensor_dependence.at(2), 1);
    EXPECT_EQ(cuda_plan.tensor_dependence.at(3), 0);
  }

  { // Check tasks
    EXPECT_EQ(cuda_plan.tasks.size(), 3);
    const auto &task0 = cuda_plan.tasks[0];
    EXPECT_EQ(task0.name, "node2");
    EXPECT_TRUE(std::holds_alternative<cuda::SiLUKernel>(task0.kernel));
    EXPECT_TRUE(task0.predecessors.empty());
    EXPECT_EQ(task0.successors, std::vector<uint32_t>{1});
    EXPECT_EQ(task0.input_descs,
              (std::vector<const TensorDesc *>{&cuda_plan.tensor_descs.at(0)}));
    EXPECT_EQ(task0.output_descs,
              (std::vector<TensorDesc *>{&cuda_plan.tensor_descs.at(2)}));
    const auto &task1 = cuda_plan.tasks[1];
    EXPECT_EQ(task1.name, "node3");
    EXPECT_TRUE(std::holds_alternative<cuda::SiLUKernel>(task1.kernel));
    EXPECT_EQ(task1.predecessors, std::vector<uint32_t>{0});
    EXPECT_TRUE(task1.successors.empty());
    EXPECT_EQ(task1.input_descs,
              (std::vector<const TensorDesc *>{&cuda_plan.tensor_descs.at(2)}));
    EXPECT_EQ(task1.output_descs,
              (std::vector<TensorDesc *>{&cuda_plan.tensor_descs.at(3)}));
    const auto &task2 = cuda_plan.tasks[2];
    EXPECT_EQ(task2.name, "node1");
    EXPECT_TRUE(std::holds_alternative<cuda::SiLUKernel>(task2.kernel));
    EXPECT_TRUE(task2.predecessors.empty());
    EXPECT_TRUE(task2.successors.empty());
    EXPECT_EQ(task2.input_descs,
              (std::vector<const TensorDesc *>{&cuda_plan.tensor_descs.at(0)}));
    EXPECT_EQ(task2.output_descs,
              (std::vector<TensorDesc *>{&cuda_plan.tensor_descs.at(1)}));
  }

  {
    cuda::CudaRuntime cuda_runtime(
        std::move(cuda_plan),
        WeightManagerWrapper(
            SafeTensorWeightManager{utils::BazelRunfile::RLocation(
                "tiny_llm/tests/datas/test.safetensors")}));
    EXPECT_TRUE(cuda_runtime.input_names() ==
                std::vector<std::string>{"input1"});
    auto output_names = cuda_runtime.output_names();
    std::ranges::sort(output_names);
    EXPECT_TRUE(
        (output_names == std::vector<std::string>{"node1_out", "node3_out"}));

    Tensor input1({.type = DeviceType::kCpu}, DataType::kFloat32, {2, 3, 4, 10},
                  true);
    EXPECT_ANY_THROW(cuda_runtime.bind_input("input1", input1));
    EXPECT_ANY_THROW(cuda_runtime.cpu_tensor_copy_to_input("input1", input1));
    input1.reallocate({2, 3, 4, 1});
    *input1.data<float>() = 1.F;
    cuda_runtime.cpu_tensor_copy_to_input("input1", input1);
    cuda_runtime.execute();

    Tensor node1_out({.type = DeviceType::kCpu}, DataType::kFloat32, {2}, true);
    cuda_runtime.output_copy_to_cpu_tensor("node1_out", node1_out);
    EXPECT_FLOAT_EQ(*node1_out.data<float>(), silu(1.F));
    EXPECT_EQ(node1_out.shape(), (std::vector<int64_t>{2, 3, 4, 1}));
    Tensor node3_out({.type = DeviceType::kCudaHost}, DataType::kFloat32,
                     {2, 3, 4, 12}, true);
    cuda_runtime.output_copy_to_cpu_tensor("node3_out", node3_out);
    EXPECT_FLOAT_EQ(*node3_out.data<float>(), silu(silu(1.F)));
    EXPECT_EQ(node3_out.shape(), (std::vector<int64_t>{2, 3, 4, 1}));
  }
}

} // namespace tiny_llm
