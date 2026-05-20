#include "tests/utils/weight_manager.hpp"
#include "tiny_llm/runtime/cuda/cuda_runtime.hpp"
#include "gtest/gtest.h"
#include <cstdint>

namespace tiny_llm {
TEST(Runtime, Linear) {
  Graph graph;
  graph.add_tensor("input", DataType::kFloat32, {1, 1, 1});
  graph.add_tensor("weight", DataType::kFloat32, {2, 3});
  graph.add_node("linear",
                 {.input_names = {"input", "weight"}, .output_names = {"out"}},
                 LinearParam{.in_dim = 3, .out_dim = 2, .bias = false});
  graph.set_input_names({"input"});
  graph.set_output_names({"out"});
  EXPECT_TRUE(is_valid(graph));

  auto cuda_plan = cuda::create_cuda_plan(
      graph,
      {.named_shape_ranges = {
           {"input", {.min_shape = {1, 1, 3}, .max_shape = {1, 4, 3}}}}});
  { // Check input infos
    EXPECT_EQ(cuda_plan.input_infos.size(), 1);
    const auto &plan_input_0 = cuda_plan.input_infos.at("input");
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
    EXPECT_EQ(cuda_plan.tensor_descs.at(0).name, "input");
    EXPECT_EQ(cuda_plan.tensor_descs.at(0).dtype, DataType::kFloat32);
    EXPECT_EQ(cuda_plan.tensor_descs.at(0).cur_shape,
              (std::vector<size_t>{1, 4, 3}));
    EXPECT_EQ(cuda_plan.tensor_descs.at(0).max_shape,
              (std::vector<size_t>{1, 4, 3}));
    EXPECT_EQ(cuda_plan.tensor_descs.at(1).name, "weight");
    EXPECT_EQ(cuda_plan.tensor_descs.at(1).dtype, DataType::kFloat32);
    EXPECT_EQ(cuda_plan.tensor_descs.at(1).cur_shape,
              (std::vector<size_t>{2, 3}));
    EXPECT_EQ(cuda_plan.tensor_descs.at(1).max_shape,
              (std::vector<size_t>{2, 3}));
    EXPECT_EQ(cuda_plan.tensor_descs.at(2).name, "out");
    EXPECT_EQ(cuda_plan.tensor_descs.at(2).dtype, DataType::kFloat32);
    EXPECT_EQ(cuda_plan.tensor_descs.at(2).cur_shape,
              (std::vector<size_t>{1, 4, 2}));
    EXPECT_EQ(cuda_plan.tensor_descs.at(2).max_shape,
              (std::vector<size_t>{1, 4, 2}));
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
    EXPECT_EQ(task0.name, "linear");
    EXPECT_TRUE(std::holds_alternative<cuda::LinearKernel>(task0.kernel));
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
    std::vector<float> weight(6);
    for (uint32_t i = 0; i < 6; ++i) {
      weight[i] = static_cast<float>(i);
    }
    wmw.set_tensor("weight", {.dtype = DataType::kFloat32,
                              .shape = {2, 3},
                              .data = weight.data(),
                              .data_len = 6 * sizeof(float)});

    cuda::CudaRuntime cuda_runtime(std::move(cuda_plan), wmw);
    EXPECT_TRUE(cuda_runtime.input_names() ==
                std::vector<std::string>{"input"});
    EXPECT_TRUE(cuda_runtime.output_names() == std::vector<std::string>{"out"});

    {
      Tensor input({.type = DeviceType::kCpu, .id = 0}, DataType::kFloat32,
                   {1, 2, 3}, true);
      {
        auto *input_ptr = input.data<float>();
        for (uint32_t i = 0; i < 6; ++i) {
          input_ptr[i] = static_cast<float>(i);
        }
      }
      cuda_runtime.cpu_tensor_copy_to_input("input", input);
      cuda_runtime.execute();

      Tensor out({.type = DeviceType::kCpu}, DataType::kFloat32, {2}, true);
      cuda_runtime.output_copy_to_cpu_tensor("out", out);
      EXPECT_EQ(out.shape(), (std::vector<int64_t>{1, 2, 2}));
      {
        const auto *out_ptr = out.data<float>();
        EXPECT_FLOAT_EQ(out_ptr[0], 5.F);
        EXPECT_FLOAT_EQ(out_ptr[1], 14.F);
        EXPECT_FLOAT_EQ(out_ptr[2], 14.F);
        EXPECT_FLOAT_EQ(out_ptr[3], 50.F);
      }
    }
  }
}

TEST(Runtime, SliceLinear) {
  Graph graph;
  graph.add_tensor("input", DataType::kFloat32, {1, 1, 1});
  graph.add_tensor("weight", DataType::kFloat32, {2, 3});
  graph.add_node(
      "slice_linear",
      {.input_names = {"input", "weight"}, .output_names = {"out"}},
      SliceLinearParam{
          .in_dim = 3, .out_dim = 2, .bias = false, .only_last_q = 3});
  graph.set_input_names({"input"});
  graph.set_output_names({"out"});
  EXPECT_TRUE(is_valid(graph));

  auto cuda_plan = cuda::create_cuda_plan(
      graph,
      {.named_shape_ranges = {
           {"input", {.min_shape = {1, 1, 3}, .max_shape = {1, 5, 3}}}}});
  { // Check input infos
    EXPECT_EQ(cuda_plan.input_infos.size(), 1);
    const auto &plan_input_0 = cuda_plan.input_infos.at("input");
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
    EXPECT_EQ(cuda_plan.tensor_descs.at(0).name, "input");
    EXPECT_EQ(cuda_plan.tensor_descs.at(0).dtype, DataType::kFloat32);
    EXPECT_EQ(cuda_plan.tensor_descs.at(0).cur_shape,
              (std::vector<size_t>{1, 5, 3}));
    EXPECT_EQ(cuda_plan.tensor_descs.at(0).max_shape,
              (std::vector<size_t>{1, 5, 3}));
    EXPECT_EQ(cuda_plan.tensor_descs.at(1).name, "weight");
    EXPECT_EQ(cuda_plan.tensor_descs.at(1).dtype, DataType::kFloat32);
    EXPECT_EQ(cuda_plan.tensor_descs.at(1).cur_shape,
              (std::vector<size_t>{2, 3}));
    EXPECT_EQ(cuda_plan.tensor_descs.at(1).max_shape,
              (std::vector<size_t>{2, 3}));
    EXPECT_EQ(cuda_plan.tensor_descs.at(2).name, "out");
    EXPECT_EQ(cuda_plan.tensor_descs.at(2).dtype, DataType::kFloat32);
    EXPECT_EQ(cuda_plan.tensor_descs.at(2).cur_shape,
              (std::vector<size_t>{1, 3, 2}));
    EXPECT_EQ(cuda_plan.tensor_descs.at(2).max_shape,
              (std::vector<size_t>{1, 3, 2}));
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
    EXPECT_EQ(task0.name, "slice_linear");
    EXPECT_TRUE(std::holds_alternative<cuda::SliceLinearKernel>(task0.kernel));
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
    std::vector<float> weight(6);
    for (uint32_t i = 0; i < 6; ++i) {
      weight[i] = static_cast<float>(i);
    }
    wmw.set_tensor("weight", {.dtype = DataType::kFloat32,
                              .shape = {2, 3},
                              .data = weight.data(),
                              .data_len = 6 * sizeof(float)});

    cuda::CudaRuntime cuda_runtime(std::move(cuda_plan), wmw);
    EXPECT_TRUE(cuda_runtime.input_names() ==
                std::vector<std::string>{"input"});
    EXPECT_TRUE(cuda_runtime.output_names() == std::vector<std::string>{"out"});

    {
      Tensor input({.type = DeviceType::kCpu, .id = 0}, DataType::kFloat32,
                   {1, 5, 3}, true);
      {
        auto *input_ptr = input.data<float>();
        for (uint32_t i = 0; i < 15; ++i) {
          input_ptr[i] = static_cast<float>(i);
        }
      }
      cuda_runtime.cpu_tensor_copy_to_input("input", input);
      cuda_runtime.execute();

      Tensor out({.type = DeviceType::kCpu}, DataType::kFloat32, {2}, true);
      cuda_runtime.output_copy_to_cpu_tensor("out", out);
      EXPECT_EQ(out.shape(), (std::vector<int64_t>{1, 3, 2}));
      {
        const auto *out_ptr = out.data<float>();
        EXPECT_FLOAT_EQ(out_ptr[0], 23.F);
        EXPECT_FLOAT_EQ(out_ptr[1], 86.F);
        EXPECT_FLOAT_EQ(out_ptr[2], 32.F);
        EXPECT_FLOAT_EQ(out_ptr[3], 122.F);
        EXPECT_FLOAT_EQ(out_ptr[4], 41.F);
        EXPECT_FLOAT_EQ(out_ptr[5], 158.F);
      }
    }
  }
}

TEST(Runtime, SliceLinearPaged) {
  Graph graph;
  graph.add_tensor("input", DataType::kFloat32, {1, 1, 1});
  graph.add_tensor("weight", DataType::kFloat32, {2, 3});
  graph.add_node("paged_slice_linear",
                 {.input_names = {"input", "weight"}, .output_names = {"out"}},
                 SliceLinearParam{.in_dim = 3,
                                  .out_dim = 2,
                                  .bias = false,
                                  .only_last_q = 1,
                                  .paged = true});
  graph.set_input_names({"input"});
  graph.set_output_names({"out"});
  EXPECT_TRUE(is_valid(graph));

  auto cuda_plan = cuda::create_cuda_plan(
      graph,
      {.named_shape_ranges = {
           {"input", {.min_shape = {1, 1, 3}, .max_shape = {1, 10, 3}}}}});
  { // Check input infos
    EXPECT_EQ(cuda_plan.input_infos.size(), 1);
    const auto &plan_input_0 = cuda_plan.input_infos.at("input");
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
    EXPECT_EQ(cuda_plan.tensor_descs.at(0).name, "input");
    EXPECT_EQ(cuda_plan.tensor_descs.at(0).dtype, DataType::kFloat32);
    EXPECT_EQ(cuda_plan.tensor_descs.at(0).cur_shape,
              (std::vector<size_t>{1, 10, 3}));
    EXPECT_EQ(cuda_plan.tensor_descs.at(0).max_shape,
              (std::vector<size_t>{1, 10, 3}));
    EXPECT_EQ(cuda_plan.tensor_descs.at(1).name, "weight");
    EXPECT_EQ(cuda_plan.tensor_descs.at(1).dtype, DataType::kFloat32);
    EXPECT_EQ(cuda_plan.tensor_descs.at(1).cur_shape,
              (std::vector<size_t>{2, 3}));
    EXPECT_EQ(cuda_plan.tensor_descs.at(1).max_shape,
              (std::vector<size_t>{2, 3}));
    EXPECT_EQ(cuda_plan.tensor_descs.at(2).name, "out");
    EXPECT_EQ(cuda_plan.tensor_descs.at(2).dtype, DataType::kFloat32);
    EXPECT_EQ(cuda_plan.tensor_descs.at(2).cur_shape,
              (std::vector<size_t>{1, 10, 2}));
    EXPECT_EQ(cuda_plan.tensor_descs.at(2).max_shape,
              (std::vector<size_t>{1, 10, 2}));
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
    EXPECT_EQ(task0.name, "paged_slice_linear");
    EXPECT_TRUE(
        std::holds_alternative<cuda::SliceLinearPagedKernel>(task0.kernel));
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
    std::vector<float> weight(6);
    for (uint32_t i = 0; i < 6; ++i) {
      weight[i] = static_cast<float>(i);
    }
    wmw.set_tensor("weight", {.dtype = DataType::kFloat32,
                              .shape = {2, 3},
                              .data = weight.data(),
                              .data_len = 6 * sizeof(float)});

    cuda::CudaRuntime cuda_runtime(std::move(cuda_plan), wmw,
                                   {.max_request_num = 3});
    EXPECT_TRUE(cuda_runtime.input_names() ==
                std::vector<std::string>{"input"});
    EXPECT_TRUE(cuda_runtime.output_names() == std::vector<std::string>{"out"});

    {
      std::vector<RequestMeta> request_metas(3);
      request_metas.at(0).seq_len = 2;
      request_metas.at(1).seq_len = 3;
      request_metas.at(2).seq_len = 5;
      Tensor input({.type = DeviceType::kCpu, .id = 0}, DataType::kFloat32,
                   {1, 10, 3}, true);
      {
        auto *input_ptr = input.data<float>();
        for (uint32_t i = 0; i < 30; ++i) {
          input_ptr[i] = static_cast<float>(i);
        }
      }

      cuda_runtime.bind_request_meta(request_metas);
      EXPECT_TRUE(request_metas[0].seq_len == 2);
      EXPECT_TRUE(request_metas[0].kv_len == 2);
      EXPECT_TRUE(request_metas[0].kv_pages.empty());
      EXPECT_TRUE(request_metas[1].seq_len == 3);
      EXPECT_TRUE(request_metas[1].kv_len == 3);
      EXPECT_TRUE(request_metas[1].kv_pages.empty());
      EXPECT_TRUE(request_metas[2].seq_len == 5);
      EXPECT_TRUE(request_metas[2].kv_len == 5);
      EXPECT_TRUE(request_metas[2].kv_pages.empty());

      cuda_runtime.cpu_tensor_copy_to_input("input", input);
      cuda_runtime.execute();

      Tensor out({.type = DeviceType::kCpu}, DataType::kFloat32, {2}, true);
      cuda_runtime.output_copy_to_cpu_tensor("out", out);
      EXPECT_EQ(out.shape(), (std::vector<int64_t>{1, 3, 2}));
      {
        const auto *out_ptr = out.data<float>();
        EXPECT_FLOAT_EQ(out_ptr[0], 14.F);
        EXPECT_FLOAT_EQ(out_ptr[1], 50.F);
        EXPECT_FLOAT_EQ(out_ptr[2], 41.F);
        EXPECT_FLOAT_EQ(out_ptr[3], 158.F);
        EXPECT_FLOAT_EQ(out_ptr[4], 86.F);
        EXPECT_FLOAT_EQ(out_ptr[5], 338.F);
      }

      cuda_runtime.destroy_request_meta(request_metas);
      EXPECT_TRUE(request_metas[0].seq_len == 0);
      EXPECT_TRUE(request_metas[0].kv_len == 0);
      EXPECT_TRUE(request_metas[0].kv_pages.empty());
      EXPECT_TRUE(request_metas[1].seq_len == 0);
      EXPECT_TRUE(request_metas[1].kv_len == 0);
      EXPECT_TRUE(request_metas[1].kv_pages.empty());
      EXPECT_TRUE(request_metas[2].seq_len == 0);
      EXPECT_TRUE(request_metas[2].kv_len == 0);
      EXPECT_TRUE(request_metas[2].kv_pages.empty());
    }
  }
}
} // namespace tiny_llm
