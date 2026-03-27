#include "tests/utils/weight_manager.hpp"
#include "tiny_llm/runtime/cuda/cuda_runtime.hpp"
#include "gtest/gtest.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace tiny_llm {
TEST(Runtime, Arithmetic) {
  Graph graph;
  graph.add_tensor("input1", DataType::kFloat32, {3});
  graph.add_node(
      "add1",
      {.input_names = {"input1", "input1"}, .output_names = {"add1_out"}},
      AddParam{.left_idx = 0, .right_idx = 0, .out_idx = 0});
  graph.add_tensor("input2", DataType::kFloat32, {3});
  graph.add_node(
      "add2",
      {.input_names = {"input2", "add1_out"}, .output_names = {"add2_out"}},
      AddParam{.left_idx = 0, .right_idx = 1, .out_idx = 1});
  graph.add_tensor("input3", DataType::kFloat32, {3});
  graph.add_node(
      "mul",
      {.input_names = {"add2_out", "input3"}, .output_names = {"mul_out"}},
      MulParam{.left_idx = 0, .right_idx = 1, .out_idx = 2});
  graph.set_input_names({"input1", "input2", "input3"});
  graph.set_output_names({"mul_out"});
  EXPECT_TRUE(is_valid(graph));

  auto cuda_plan = cuda::create_cuda_plan(
      graph, {.named_shape_ranges = {
                  {"input1", {.min_shape = {1}, .max_shape = {10}}},
                  {"input2", {.min_shape = {1}, .max_shape = {10}}},
                  {"input3", {.min_shape = {1}, .max_shape = {10}}}}});
  { // Check input infos
    EXPECT_EQ(cuda_plan.input_infos.size(), 3);
    EXPECT_EQ(std::get<0>(cuda_plan.input_infos.at("input1")), 0);
    const auto &plan_input1_task_infos =
        std::get<1>(cuda_plan.input_infos.at("input1"));
    EXPECT_EQ(plan_input1_task_infos.size(), 2);
    EXPECT_EQ(plan_input1_task_infos.at(0).task_id, 0);
    EXPECT_EQ(plan_input1_task_infos.at(0).io_id, 0);
    EXPECT_EQ(plan_input1_task_infos.at(1).task_id, 0);
    EXPECT_EQ(plan_input1_task_infos.at(1).io_id, 1);
    EXPECT_EQ(std::get<0>(cuda_plan.input_infos.at("input2")), 2);
    const auto &plan_input2_task_infos =
        std::get<1>(cuda_plan.input_infos.at("input2"));
    EXPECT_EQ(plan_input2_task_infos.size(), 1);
    EXPECT_EQ(plan_input2_task_infos.at(0).task_id, 1);
    EXPECT_EQ(plan_input2_task_infos.at(0).io_id, 0);
    EXPECT_EQ(std::get<0>(cuda_plan.input_infos.at("input3")), 4);
    const auto &plan_input3_task_infos =
        std::get<1>(cuda_plan.input_infos.at("input3"));
    EXPECT_EQ(plan_input3_task_infos.size(), 1);
    EXPECT_EQ(plan_input3_task_infos.at(0).task_id, 2);
    EXPECT_EQ(plan_input3_task_infos.at(0).io_id, 1);
  }

  { // Check output infos
    EXPECT_EQ(cuda_plan.output_infos.size(), 1);
    EXPECT_EQ(std::get<0>(cuda_plan.output_infos.at("mul_out")), 5);
    EXPECT_EQ(std::get<1>(cuda_plan.output_infos.at("mul_out")).task_id, 2);
    EXPECT_EQ(std::get<1>(cuda_plan.output_infos.at("mul_out")).io_id, 0);
  }

  {
    // Check descs
    EXPECT_EQ(cuda_plan.tensor_descs.size(), 6);
    EXPECT_EQ(cuda_plan.tensor_descs.at(0).name, "input1");
    EXPECT_EQ(cuda_plan.tensor_descs.at(1).name, "add1_out");
    EXPECT_EQ(cuda_plan.tensor_descs.at(2).name, "input2");
    EXPECT_EQ(cuda_plan.tensor_descs.at(3).name, "add2_out");
    EXPECT_EQ(cuda_plan.tensor_descs.at(4).name, "input3");
    EXPECT_EQ(cuda_plan.tensor_descs.at(5).name, "mul_out");
    for (uint32_t i = 0; i < 6; ++i) {
      const auto &cur_desc = cuda_plan.tensor_descs.at(i);
      EXPECT_EQ(cur_desc.dtype, DataType::kFloat32);
      EXPECT_EQ(cur_desc.cur_shape, std::vector<size_t>{10});
      EXPECT_EQ(cur_desc.max_shape, std::vector<size_t>{10});
    }
  }

  {
    // Check dependence
    EXPECT_EQ(cuda_plan.tensor_dependence.size(), 6);
    EXPECT_EQ(cuda_plan.tensor_dependence.at(0), 2);
    EXPECT_EQ(cuda_plan.tensor_dependence.at(1), 1);
    EXPECT_EQ(cuda_plan.tensor_dependence.at(2), 1);
    EXPECT_EQ(cuda_plan.tensor_dependence.at(3), 1);
    EXPECT_EQ(cuda_plan.tensor_dependence.at(4), 1);
    EXPECT_EQ(cuda_plan.tensor_dependence.at(5), 0);
  }

  { // Check tasks
    EXPECT_EQ(cuda_plan.tasks.size(), 3);
    {
      const auto &task0 = cuda_plan.tasks[0];
      EXPECT_EQ(task0.name, "add1");
      EXPECT_TRUE(std::holds_alternative<cuda::AddKernel>(task0.kernel));
      EXPECT_TRUE(task0.predecessors.empty());
      EXPECT_TRUE(task0.successors == std::vector<uint32_t>{1});
      EXPECT_EQ(task0.input_descs, (std::vector<const TensorDesc *>{
                                       &cuda_plan.tensor_descs.at(0),
                                       &cuda_plan.tensor_descs.at(0)}));
      EXPECT_EQ(task0.output_descs,
                (std::vector<TensorDesc *>{&cuda_plan.tensor_descs.at(1)}));
    }
    {
      const auto &task1 = cuda_plan.tasks[1];
      EXPECT_EQ(task1.name, "add2");
      EXPECT_TRUE(std::holds_alternative<cuda::AddKernel>(task1.kernel));
      EXPECT_TRUE(task1.predecessors == std::vector<uint32_t>{0});
      EXPECT_TRUE(task1.successors == std::vector<uint32_t>{2});
      EXPECT_EQ(task1.input_descs, (std::vector<const TensorDesc *>{
                                       &cuda_plan.tensor_descs.at(2),
                                       &cuda_plan.tensor_descs.at(1)}));
      EXPECT_EQ(task1.output_descs,
                (std::vector<TensorDesc *>{&cuda_plan.tensor_descs.at(3)}));
    }
    {
      const auto &task2 = cuda_plan.tasks[2];
      EXPECT_EQ(task2.name, "mul");
      EXPECT_TRUE(std::holds_alternative<cuda::MulKernel>(task2.kernel));
      EXPECT_TRUE(task2.predecessors == std::vector<uint32_t>{1});
      EXPECT_TRUE(task2.successors.empty());
      EXPECT_EQ(task2.input_descs, (std::vector<const TensorDesc *>{
                                       &cuda_plan.tensor_descs.at(3),
                                       &cuda_plan.tensor_descs.at(4)}));
      EXPECT_EQ(task2.output_descs,
                (std::vector<TensorDesc *>{&cuda_plan.tensor_descs.at(5)}));
    }
  }

  {
    cuda::CudaRuntime cuda_runtime(std::move(cuda_plan),
                                   WeightManagerWrapper{TestWeightManager{}});
    auto input_names = cuda_runtime.input_names();
    std::ranges::sort(input_names);
    EXPECT_TRUE(input_names ==
                (std::vector<std::string>{"input1", "input2", "input3"}));
    EXPECT_TRUE(cuda_runtime.output_names() ==
                std::vector<std::string>{"mul_out"});

    Tensor input1({.type = DeviceType::kCpu}, DataType::kFloat32, {10});
    Tensor input2({.type = DeviceType::kCpu}, DataType::kFloat32, {10});
    Tensor input3({.type = DeviceType::kCpu}, DataType::kFloat32, {10});
    {
      auto *input1_ptr = input1.data<float>();
      auto *input2_ptr = input2.data<float>();
      auto *input3_ptr = input3.data<float>();
      for (size_t i = 0; i < 10; ++i) {
        input1_ptr[i] = 1.F;
        input2_ptr[i] = 3.F;
        input3_ptr[i] = 4.F;
      }
    }
    input1 = input1.to({.type = DeviceType::kCuda, .id = 0});
    input2 = input2.to({.type = DeviceType::kCuda, .id = 0});
    ThreadCudaContexts::Synchronize();

    cuda_runtime.bind_input("input1", input1);
    cuda_runtime.bind_input("input2", input2);
    cuda_runtime.cpu_tensor_copy_to_input("input3", input3);
    cuda_runtime.execute();

    Tensor mul_out({.type = DeviceType::kCpu}, DataType::kFloat32, {2}, true);
    cuda_runtime.output_copy_to_cpu_tensor("mul_out", mul_out);
    EXPECT_EQ(mul_out.shape(), (std::vector<int64_t>{10}));
    const auto *mul_out_ptr = mul_out.data<float>();
    for (size_t i = 0; i < 10; ++i) {
      EXPECT_FLOAT_EQ(mul_out_ptr[i], 20.F);
    }

    Tensor bad_input3({.type = DeviceType::kCpu}, DataType::kFloat32, {9},
                      true);
    cuda_runtime.cpu_tensor_copy_to_input("input3", bad_input3);
    EXPECT_ANY_THROW(cuda_runtime.execute());
    // sync
    cuda_runtime.output_copy_to_cpu_tensor("mul_out", mul_out);
  }
}
} // namespace tiny_llm
