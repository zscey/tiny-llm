#include "tiny_llm/runtime/cuda/cuda_runtime.hpp"
#include "tiny_llm/utils/runfile.hpp"
#include "tiny_llm/weight_managers/safetensors/weight_manager.hpp"
#include "gtest/gtest.h"
#include <algorithm>
#include <cstdint>

namespace tiny_llm {
TEST(Runtime, Rope) {
  Graph graph;
  graph.add_node(
      "rope", {.output_names = {"cos", "sin"}},
      RopeParam{.max_len = 2, .head_dim = 96, .theta = 10000., .pin = true});

  graph.set_output_names({"cos", "sin"});
  EXPECT_TRUE(is_valid(graph));

  auto cuda_plan = cuda::create_cuda_plan(graph, {});
  { // Check input infos
    EXPECT_EQ(cuda_plan.input_infos.size(), 0);
  }

  { // Check output infos
    EXPECT_EQ(cuda_plan.output_infos.size(), 2);
    EXPECT_EQ(std::get<0>(cuda_plan.output_infos.at("cos")), 0);
    EXPECT_EQ(std::get<1>(cuda_plan.output_infos.at("cos")).task_id, 0);
    EXPECT_EQ(std::get<1>(cuda_plan.output_infos.at("cos")).io_id, 0);
    EXPECT_EQ(std::get<0>(cuda_plan.output_infos.at("sin")), 1);
    EXPECT_EQ(std::get<1>(cuda_plan.output_infos.at("sin")).task_id, 0);
    EXPECT_EQ(std::get<1>(cuda_plan.output_infos.at("sin")).io_id, 1);
  }

  {
    // Check descs
    EXPECT_EQ(cuda_plan.tensor_descs.size(), 2);
    EXPECT_EQ(cuda_plan.tensor_descs.at(0).name, "cos");
    EXPECT_EQ(cuda_plan.tensor_descs.at(0).dtype, DataType::kFloat32);
    EXPECT_EQ(cuda_plan.tensor_descs.at(0).cur_shape,
              (std::vector<size_t>{2, 48}));
    EXPECT_EQ(cuda_plan.tensor_descs.at(0).max_shape,
              (std::vector<size_t>{2, 48}));
    EXPECT_EQ(cuda_plan.tensor_descs.at(1).name, "sin");
    EXPECT_EQ(cuda_plan.tensor_descs.at(1).dtype, DataType::kFloat32);
    EXPECT_EQ(cuda_plan.tensor_descs.at(1).cur_shape,
              (std::vector<size_t>{2, 48}));
    EXPECT_EQ(cuda_plan.tensor_descs.at(1).max_shape,
              (std::vector<size_t>{2, 48}));
  }

  {
    // Check dependence
    EXPECT_EQ(cuda_plan.tensor_dependence.size(), 2);
    EXPECT_EQ(cuda_plan.tensor_dependence.at(0), 0);
    EXPECT_EQ(cuda_plan.tensor_dependence.at(1), 0);
  }

  { // Check tasks
    EXPECT_EQ(cuda_plan.tasks.size(), 1);
    const auto &task0 = cuda_plan.tasks[0];
    EXPECT_EQ(task0.name, "rope");
    EXPECT_TRUE(std::holds_alternative<cuda::RopeKernel>(task0.kernel));
    EXPECT_TRUE(task0.predecessors.empty());
    EXPECT_TRUE(task0.successors.empty());
    EXPECT_TRUE(task0.input_descs.empty());
    EXPECT_EQ(task0.output_descs,
              (std::vector<TensorDesc *>{&cuda_plan.tensor_descs.at(0),
                                         &cuda_plan.tensor_descs.at(1)}));
  }

  {
    WeightManagerWrapper wmw(
        SafeTensorWeightManager(utils::BazelRunfile::RLocation(
            "tiny_llm/tests/datas/rope_s2_d96.safetensors")));
    cuda::CudaRuntime cuda_runtime(std::move(cuda_plan), wmw);
    EXPECT_TRUE(cuda_runtime.input_names().empty());
    auto output_names = cuda_runtime.output_names();
    std::ranges::sort(output_names);
    EXPECT_TRUE((output_names == std::vector<std::string>{"cos", "sin"}));
    {
      cuda_runtime.execute();
      cuda_runtime.execute();

      Tensor cos({.type = DeviceType::kCpu}, DataType::kFloat32, {2}, true);
      Tensor sin({.type = DeviceType::kCpu}, DataType::kFloat32, {2, 3}, true);
      cuda_runtime.output_copy_to_cpu_tensor("cos", cos);
      cuda_runtime.output_copy_to_cpu_tensor("sin", sin);
      EXPECT_EQ(cos.shape(), (std::vector<int64_t>{2, 48}));
      EXPECT_EQ(sin.shape(), (std::vector<int64_t>{2, 48}));
      const auto *cos_target_ptr =
          static_cast<const float *>(wmw.get_tensor("cos").data);
      const auto *cos_ptr = cos.data<float>();
      const auto *sin_target_ptr =
          static_cast<const float *>(wmw.get_tensor("sin").data);
      const auto *sin_ptr = sin.data<float>();
      for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 48; ++j) {
          auto dst_shift = (i * 48) + j;
          auto target_shift = (i * 96) + j;
          EXPECT_FLOAT_EQ(cos_ptr[dst_shift], cos_target_ptr[target_shift]);
          EXPECT_FLOAT_EQ(sin_ptr[dst_shift], sin_target_ptr[target_shift]);
        }
      }
    }
  }
}
} // namespace tiny_llm
