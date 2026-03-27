#include "tests/utils/weight_manager.hpp"
#include "tiny_llm/runtime/cuda/cuda_runtime.hpp"
#include "tiny_llm/utils/runfile.hpp"
#include "tiny_llm/weight_managers/safetensors/weight_manager.hpp"
#include "gtest/gtest.h"
#include <algorithm>
#include <cstdint>

namespace tiny_llm {
TEST(Runtime, RMSNorm) {
  Graph graph;
  graph.add_tensor("input", DataType::kFloat32, {2, 107, 241});
  graph.add_tensor("rmsnorm.weight", DataType::kFloat32, {241});
  graph.add_node(
      "rmsnorm",
      {.input_names = {"input", "rmsnorm.weight"}, .output_names = {"out"}},
      RMSNormParam{.hidden_size = 241, .eps = 1.192e-7F, .inplace = true});
  graph.set_input_names({"input"});
  graph.set_output_names({"out"});
  EXPECT_TRUE(is_valid(graph));

  auto cuda_plan = cuda::create_cuda_plan(
      graph,
      {.named_shape_ranges = {
           {"input", {.min_shape = {1, 2, 241}, .max_shape = {2, 128, 241}}}}});
  { // Check input infos
    EXPECT_EQ(cuda_plan.input_infos.size(), 1);
    EXPECT_EQ(std::get<0>(cuda_plan.input_infos.at("input")), 0);
    const auto &plan_input_0_task_infos =
        std::get<1>(cuda_plan.input_infos.at("input"));
    EXPECT_EQ(plan_input_0_task_infos.size(), 1);
    EXPECT_EQ(plan_input_0_task_infos.at(0).task_id, 0);
    EXPECT_EQ(plan_input_0_task_infos.at(0).io_id, 0);
  }

  { // Check output infos
    EXPECT_EQ(cuda_plan.output_infos.size(), 1);
    EXPECT_EQ(std::get<0>(cuda_plan.output_infos.at("out")), 2);
    EXPECT_EQ(std::get<1>(cuda_plan.output_infos.at("out")).task_id, 0);
    EXPECT_EQ(std::get<1>(cuda_plan.output_infos.at("out")).io_id, 0);
  }

  {
    // Check descs
    EXPECT_EQ(cuda_plan.tensor_descs.size(), 3);
    EXPECT_EQ(cuda_plan.tensor_descs.at(0).name, "input");
    EXPECT_EQ(cuda_plan.tensor_descs.at(0).dtype, DataType::kFloat32);
    EXPECT_EQ(cuda_plan.tensor_descs.at(0).cur_shape,
              (std::vector<size_t>{2, 128, 241}));
    EXPECT_EQ(cuda_plan.tensor_descs.at(0).max_shape,
              (std::vector<size_t>{2, 128, 241}));
    EXPECT_EQ(cuda_plan.tensor_descs.at(1).name, "rmsnorm.weight");
    EXPECT_EQ(cuda_plan.tensor_descs.at(1).dtype, DataType::kFloat32);
    EXPECT_EQ(cuda_plan.tensor_descs.at(1).cur_shape,
              (std::vector<size_t>{241}));
    EXPECT_EQ(cuda_plan.tensor_descs.at(1).max_shape,
              (std::vector<size_t>{241}));
    EXPECT_EQ(cuda_plan.tensor_descs.at(2).name, "out");
    EXPECT_EQ(cuda_plan.tensor_descs.at(2).dtype, DataType::kFloat32);
    EXPECT_EQ(cuda_plan.tensor_descs.at(2).cur_shape,
              (std::vector<size_t>{2, 128, 241}));
    EXPECT_EQ(cuda_plan.tensor_descs.at(2).max_shape,
              (std::vector<size_t>{2, 128, 241}));
  }

  {
    // Check dependence
    EXPECT_EQ(cuda_plan.tensor_dependence.size(), 3);
    EXPECT_EQ(cuda_plan.tensor_dependence.at(0), 1);
    EXPECT_EQ(cuda_plan.tensor_dependence.at(1), 1);
    EXPECT_EQ(cuda_plan.tensor_dependence.at(2), 0);
  }

  { // Check tasks
    EXPECT_EQ(cuda_plan.tasks.size(), 1);
    const auto &task0 = cuda_plan.tasks[0];
    EXPECT_EQ(task0.name, "rmsnorm");
    EXPECT_TRUE(std::holds_alternative<cuda::RMSNormKernel>(task0.kernel));
    EXPECT_TRUE(task0.predecessors.empty());
    EXPECT_TRUE(task0.successors.empty());
    EXPECT_EQ(task0.input_descs,
              (std::vector<const TensorDesc *>{&cuda_plan.tensor_descs.at(0),
                                               &cuda_plan.tensor_descs.at(1)}));
    EXPECT_EQ(task0.output_descs,
              (std::vector<TensorDesc *>{&cuda_plan.tensor_descs.at(2)}));
  }

  {
    WeightManagerWrapper wmw(TestWeightManager{});
    {
      Tensor weight({.type = DeviceType::kCpu}, DataType::kFloat32, {241});
      auto *weight_ptr = weight.data<float>();
      for (size_t i = 0; i < 241; ++i) {
        weight_ptr[i] = static_cast<float>(i % 100) / 100.F;
      }
      wmw.set_tensor("rmsnorm.weight", {.dtype = weight.dtype(),
                                        .shape = weight.shape(),
                                        .data = weight.data(),
                                        .data_len = sizeof(float) * 241});
    }
    cuda::CudaRuntime cuda_runtime(std::move(cuda_plan), wmw);
    EXPECT_TRUE(cuda_runtime.input_names() ==
                std::vector<std::string>{"input"});
    EXPECT_TRUE(cuda_runtime.output_names() == std::vector<std::string>{"out"});
    {
      Tensor input({.type = DeviceType::kCpu}, DataType::kFloat32,
                   {2, 107, 241});
      {
        auto *input_ptr = input.data<float>();
        for (size_t i = 0, i_end = static_cast<size_t>(2 * 107 * 241);
             i < i_end; ++i) {
          input_ptr[i] = static_cast<float>(i % 100) / 100.F;
        }
        input = input.to({.type = DeviceType::kCuda, .id = 0});
      }
      cuda_runtime.bind_input("input", input);
      ThreadCudaContexts::Synchronize();
      cuda_runtime.execute();

      Tensor out({.type = DeviceType::kCpu}, DataType::kFloat32, {2}, true);
      cuda_runtime.output_copy_to_cpu_tensor("out", out);
      EXPECT_EQ(out.shape(), (std::vector<int64_t>{2, 107, 241}));

      SafeTensorWeightManager wm(utils::BazelRunfile::RLocation(
          "tiny_llm/tests/datas/rms_norm_b2l107d241.safetensors"));
      const auto *out_ptr = out.data<float>();
      const auto *target_ptr =
          reinterpret_cast<const float *>(wm.get_tensor("res").data);
      for (size_t i = 0, i_end = static_cast<size_t>(2 * 107 * 241); i < i_end;
           ++i) {
        EXPECT_TRUE(std::abs(out_ptr[i] - target_ptr[i]) < 1e-5F);
      }
    }
    {
      Tensor input({.type = DeviceType::kCpu, .id = 0}, DataType::kFloat32,
                   {1, 2, 241}, true);
      cuda_runtime.cpu_tensor_copy_to_input("input", input);
      cuda_runtime.execute();
      Tensor out({.type = DeviceType::kCpu}, DataType::kFloat32, {2}, true);
      cuda_runtime.output_copy_to_cpu_tensor("out", out);
      EXPECT_EQ(out.shape(), (std::vector<int64_t>{1, 2, 241}));
    }
  }
}
} // namespace tiny_llm
