#include "tests/utils/weight_manager.hpp"
#include "tiny_llm/runtime/cuda/cuda_runtime.hpp"
#include "gtest/gtest.h"
#include <cstdint>

namespace tiny_llm {
TEST(Runtime, Embedding) {
  Graph graph;
  graph.add_tensor("ids", DataType::kUint32, {1, 2});
  graph.add_tensor("emb.weight", DataType::kFloat32, {5, 4});
  graph.add_node(
      "emb", {.input_names = {"ids", "emb.weight"}, .output_names = {"out"}},
      EmbeddingParam{.num_embeddings = 5, .hidden_size = 4});
  graph.set_input_names({"ids"});
  graph.set_output_names({"out"});
  EXPECT_TRUE(is_valid(graph));

  auto cuda_plan = cuda::create_cuda_plan(
      graph, {.named_shape_ranges = {
                  {"ids", {.min_shape = {1, 1}, .max_shape = {1, 4}}}}});
  { // Check input infos
    EXPECT_EQ(cuda_plan.input_infos.size(), 1);
    const auto &plan_input_0 = cuda_plan.input_infos.at("ids");
    EXPECT_EQ(std::get<0>(plan_input_0), 0);
    const auto &plan_input_0_task_infos = std::get<1>(plan_input_0);
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
    EXPECT_EQ(cuda_plan.tensor_descs.at(0).name, "ids");
    EXPECT_EQ(cuda_plan.tensor_descs.at(0).dtype, DataType::kUint32);
    EXPECT_EQ(cuda_plan.tensor_descs.at(0).cur_shape,
              (std::vector<size_t>{1, 4}));
    EXPECT_EQ(cuda_plan.tensor_descs.at(0).max_shape,
              (std::vector<size_t>{1, 4}));
    EXPECT_EQ(cuda_plan.tensor_descs.at(1).name, "emb.weight");
    EXPECT_EQ(cuda_plan.tensor_descs.at(1).dtype, DataType::kFloat32);
    EXPECT_EQ(cuda_plan.tensor_descs.at(1).cur_shape,
              (std::vector<size_t>{5, 4}));
    EXPECT_EQ(cuda_plan.tensor_descs.at(1).max_shape,
              (std::vector<size_t>{5, 4}));
    EXPECT_EQ(cuda_plan.tensor_descs.at(2).name, "out");
    EXPECT_EQ(cuda_plan.tensor_descs.at(2).dtype, DataType::kFloat32);
    EXPECT_EQ(cuda_plan.tensor_descs.at(2).cur_shape,
              (std::vector<size_t>{1, 4, 4}));
    EXPECT_EQ(cuda_plan.tensor_descs.at(2).max_shape,
              (std::vector<size_t>{1, 4, 4}));
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
    EXPECT_EQ(task0.name, "emb");
    EXPECT_TRUE(std::holds_alternative<cuda::EmbeddingKernel>(task0.kernel));
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
    std::vector<float> emb_weight(20);
    for (uint32_t i = 0; i < 20; ++i) {
      emb_weight[i] = static_cast<float>(i);
    }
    wmw.set_tensor("emb.weight", {.dtype = DataType::kFloat32,
                                  .shape = {5, 4},
                                  .data = emb_weight.data(),
                                  .data_len = 20 * sizeof(float)});

    cuda::CudaRuntime cuda_runtime(std::move(cuda_plan), wmw);
    EXPECT_TRUE(cuda_runtime.input_names() == std::vector<std::string>{"ids"});
    auto output_names = cuda_runtime.output_names();
    std::ranges::sort(output_names);
    EXPECT_TRUE((output_names == std::vector<std::string>{"out"}));

    {
      Tensor ids({.type = DeviceType::kCpu, .id = 0}, DataType::kUint32, {1, 2},
                 true);
      {
        auto *ids_ptr = ids.data<uint32_t>();
        ids_ptr[0] = 3;
        ids_ptr[1] = 1;
        ids = ids.to({.type = DeviceType::kCuda, .id = 0});
        ThreadCudaContexts::Synchronize();
      }
      cuda_runtime.bind_input("ids", ids);
      cuda_runtime.execute();

      Tensor out({.type = DeviceType::kCpu}, DataType::kFloat32, {2}, true);
      cuda_runtime.output_copy_to_cpu_tensor("out", out);
      EXPECT_EQ(out.shape(), (std::vector<int64_t>{1, 2, 4}));
      {
        const auto *out_ptr = out.data<float>();
        for (size_t i = 0; i < 4; ++i) {
          EXPECT_FLOAT_EQ(out_ptr[i], static_cast<float>(i) + 12.F);
          EXPECT_FLOAT_EQ(out_ptr[i + 4], static_cast<float>(i) + 4.F);
        }
      }
    }
    {
      Tensor ids({.type = DeviceType::kCpu, .id = 0}, DataType::kUint32, {1, 2},
                 true);
      {
        auto *ids_ptr = ids.data<uint32_t>();
        ids_ptr[0] = 2;
        ids_ptr[1] = 0;
      }
      cuda_runtime.cpu_tensor_copy_to_input("ids", ids);
      cuda_runtime.execute();

      Tensor out({.type = DeviceType::kCpu}, DataType::kFloat32, {2}, true);
      cuda_runtime.output_copy_to_cpu_tensor("out", out);
      EXPECT_EQ(out.shape(), (std::vector<int64_t>{1, 2, 4}));
      {
        const auto *out_ptr = out.data<float>();
        for (size_t i = 0; i < 4; ++i) {
          EXPECT_FLOAT_EQ(out_ptr[i], static_cast<float>(i) + 8.F);
          EXPECT_FLOAT_EQ(out_ptr[i + 4], static_cast<float>(i));
        }
      }
    }
  }
}
} // namespace tiny_llm
