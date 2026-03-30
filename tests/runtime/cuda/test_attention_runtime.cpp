#include "tests/utils/weight_manager.hpp"
#include "tiny_llm/runtime/cuda/cuda_runtime.hpp"
#include "tiny_llm/utils/runfile.hpp"
#include "tiny_llm/weight_managers/safetensors/weight_manager.hpp"
#include "gtest/gtest.h"
#include <cstdint>
#include <fcntl.h>

namespace tiny_llm {
TEST(Runtime, Attention) {
  Graph graph;
  graph.add_tensor("hidden_state", DataType::kFloat32, {1, 2, 2048});
  graph.add_tensor("cos", DataType::kFloat32, {1, 5, 32});
  graph.add_tensor("sin", DataType::kFloat32, {1, 5, 32});
  graph.add_tensor("pos_ids", DataType::kUint32, {1, 2});
  graph.add_tensor("q_weight", DataType::kFloat32, {2048, 2048});
  graph.add_tensor("k_weight", DataType::kFloat32, {256, 2048});
  graph.add_tensor("v_weight", DataType::kFloat32, {256, 2048});
  graph.add_tensor("o_weight", DataType::kFloat32, {2048, 2048});
  graph.add_node(
      "attn",
      {.input_names = {"hidden_state", "cos", "sin", "pos_ids", "q_weight",
                       "k_weight", "v_weight", "o_weight"},
       .output_names = {"out"}},
      CausalAttentionParam{.head_dim = 64,
                           .q_head = 32,
                           .kv_head = 4,
                           .bias = false,
                           .max_len = 5});
  graph.set_input_names({"hidden_state", "cos", "sin", "pos_ids"});
  graph.set_output_names({"out"});
  EXPECT_TRUE(is_valid(graph));

  auto cuda_plan = cuda::create_cuda_plan(
      graph, {.named_shape_ranges = {
                  {"hidden_state",
                   {.min_shape = {1, 1, 2048}, .max_shape = {1, 5, 2048}}},
                  {"cos", {.min_shape = {5, 32}, .max_shape = {5, 32}}},
                  {"sin", {.min_shape = {5, 32}, .max_shape = {5, 32}}},
                  {"pos_ids", {.min_shape = {1, 1}, .max_shape = {1, 5}}}}});
  { // Check input infos
    EXPECT_EQ(cuda_plan.input_infos.size(), 4);
    const auto &plan_input_0 = cuda_plan.input_infos.at("hidden_state");
    EXPECT_EQ(std::get<0>(plan_input_0), 0);
    const auto &plan_input_0_task_infos = std::get<1>(plan_input_0);
    EXPECT_EQ(plan_input_0_task_infos.size(), 1);
    EXPECT_EQ(plan_input_0_task_infos.at(0).task_id, 0);
    EXPECT_EQ(plan_input_0_task_infos.at(0).io_id, 0);
    const auto &plan_input_1 = cuda_plan.input_infos.at("cos");
    EXPECT_EQ(std::get<0>(plan_input_1), 1);
    const auto &plan_input_1_task_infos = std::get<1>(plan_input_1);
    EXPECT_EQ(plan_input_1_task_infos.size(), 1);
    EXPECT_EQ(plan_input_1_task_infos.at(0).task_id, 0);
    EXPECT_EQ(plan_input_1_task_infos.at(0).io_id, 1);
    const auto &plan_input_2 = cuda_plan.input_infos.at("sin");
    EXPECT_EQ(std::get<0>(plan_input_2), 2);
    const auto &plan_input_2_task_infos = std::get<1>(plan_input_2);
    EXPECT_EQ(plan_input_2_task_infos.size(), 1);
    EXPECT_EQ(plan_input_2_task_infos.at(0).task_id, 0);
    EXPECT_EQ(plan_input_2_task_infos.at(0).io_id, 2);
    const auto &plan_input_3 = cuda_plan.input_infos.at("pos_ids");
    EXPECT_EQ(std::get<0>(plan_input_3), 3);
    const auto &plan_input_3_task_infos = std::get<1>(plan_input_3);
    EXPECT_EQ(plan_input_3_task_infos.size(), 1);
    EXPECT_EQ(plan_input_3_task_infos.at(0).task_id, 0);
    EXPECT_EQ(plan_input_3_task_infos.at(0).io_id, 3);
  }

  { // Check output infos
    EXPECT_EQ(cuda_plan.output_infos.size(), 1);
    EXPECT_EQ(std::get<0>(cuda_plan.output_infos.at("out")), 8);
    EXPECT_EQ(std::get<1>(cuda_plan.output_infos.at("out")).task_id, 0);
    EXPECT_EQ(std::get<1>(cuda_plan.output_infos.at("out")).io_id, 0);
  }

  {
    // Check descs
    EXPECT_EQ(cuda_plan.tensor_descs.size(), 9);
    EXPECT_EQ(cuda_plan.tensor_descs.at(0).name, "hidden_state");
    EXPECT_EQ(cuda_plan.tensor_descs.at(0).dtype, DataType::kFloat32);
    EXPECT_EQ(cuda_plan.tensor_descs.at(0).cur_shape,
              (std::vector<size_t>{1, 5, 2048}));
    EXPECT_EQ(cuda_plan.tensor_descs.at(0).max_shape,
              (std::vector<size_t>{1, 5, 2048}));
    EXPECT_EQ(cuda_plan.tensor_descs.at(1).name, "cos");
    EXPECT_EQ(cuda_plan.tensor_descs.at(1).dtype, DataType::kFloat32);
    EXPECT_EQ(cuda_plan.tensor_descs.at(1).cur_shape,
              (std::vector<size_t>{5, 32}));
    EXPECT_EQ(cuda_plan.tensor_descs.at(1).max_shape,
              (std::vector<size_t>{5, 32}));
    EXPECT_EQ(cuda_plan.tensor_descs.at(2).name, "sin");
    EXPECT_EQ(cuda_plan.tensor_descs.at(2).dtype, DataType::kFloat32);
    EXPECT_EQ(cuda_plan.tensor_descs.at(2).cur_shape,
              (std::vector<size_t>{5, 32}));
    EXPECT_EQ(cuda_plan.tensor_descs.at(2).max_shape,
              (std::vector<size_t>{5, 32}));
    EXPECT_EQ(cuda_plan.tensor_descs.at(3).name, "pos_ids");
    EXPECT_EQ(cuda_plan.tensor_descs.at(3).dtype, DataType::kUint32);
    EXPECT_EQ(cuda_plan.tensor_descs.at(3).cur_shape,
              (std::vector<size_t>{1, 5}));
    EXPECT_EQ(cuda_plan.tensor_descs.at(3).max_shape,
              (std::vector<size_t>{1, 5}));
    EXPECT_EQ(cuda_plan.tensor_descs.at(4).name, "q_weight");
    EXPECT_EQ(cuda_plan.tensor_descs.at(4).dtype, DataType::kFloat32);
    EXPECT_EQ(cuda_plan.tensor_descs.at(4).cur_shape,
              (std::vector<size_t>{2048, 2048}));
    EXPECT_EQ(cuda_plan.tensor_descs.at(4).max_shape,
              (std::vector<size_t>{2048, 2048}));
    EXPECT_EQ(cuda_plan.tensor_descs.at(5).name, "k_weight");
    EXPECT_EQ(cuda_plan.tensor_descs.at(5).dtype, DataType::kFloat32);
    EXPECT_EQ(cuda_plan.tensor_descs.at(5).cur_shape,
              (std::vector<size_t>{256, 2048}));
    EXPECT_EQ(cuda_plan.tensor_descs.at(5).max_shape,
              (std::vector<size_t>{256, 2048}));
    EXPECT_EQ(cuda_plan.tensor_descs.at(6).name, "v_weight");
    EXPECT_EQ(cuda_plan.tensor_descs.at(6).dtype, DataType::kFloat32);
    EXPECT_EQ(cuda_plan.tensor_descs.at(6).cur_shape,
              (std::vector<size_t>{256, 2048}));
    EXPECT_EQ(cuda_plan.tensor_descs.at(6).max_shape,
              (std::vector<size_t>{256, 2048}));
    EXPECT_EQ(cuda_plan.tensor_descs.at(7).name, "o_weight");
    EXPECT_EQ(cuda_plan.tensor_descs.at(7).dtype, DataType::kFloat32);
    EXPECT_EQ(cuda_plan.tensor_descs.at(7).cur_shape,
              (std::vector<size_t>{2048, 2048}));
    EXPECT_EQ(cuda_plan.tensor_descs.at(7).max_shape,
              (std::vector<size_t>{2048, 2048}));
    EXPECT_EQ(cuda_plan.tensor_descs.at(8).name, "out");
    EXPECT_EQ(cuda_plan.tensor_descs.at(8).dtype, DataType::kFloat32);
    EXPECT_EQ(cuda_plan.tensor_descs.at(8).cur_shape,
              (std::vector<size_t>{1, 5, 2048}));
    EXPECT_EQ(cuda_plan.tensor_descs.at(8).max_shape,
              (std::vector<size_t>{1, 5, 2048}));
  }

  {
    // Check dependence
    EXPECT_EQ(cuda_plan.tensor_dependence.size(), 9);
    for (uint32_t i = 0; i < 8; ++i) {
      EXPECT_EQ(cuda_plan.tensor_dependence.at(i), 1);
    }
    EXPECT_EQ(cuda_plan.tensor_dependence.at(8), 0);
  }

  { // Check tasks
    EXPECT_EQ(cuda_plan.tasks.size(), 1);
    const auto &task0 = cuda_plan.tasks[0];
    EXPECT_EQ(task0.name, "attn");
    EXPECT_TRUE(
        std::holds_alternative<cuda::CausalAttentionKernel>(task0.kernel));
    EXPECT_TRUE(task0.predecessors.empty());
    EXPECT_TRUE(task0.successors.empty());
    EXPECT_EQ(
        task0.input_descs,
        (std::vector<const TensorDesc *>{
            &cuda_plan.tensor_descs.at(0), &cuda_plan.tensor_descs.at(1),
            &cuda_plan.tensor_descs.at(2), &cuda_plan.tensor_descs.at(3),
            &cuda_plan.tensor_descs.at(4), &cuda_plan.tensor_descs.at(5),
            &cuda_plan.tensor_descs.at(6), &cuda_plan.tensor_descs.at(7)}));
    EXPECT_EQ(task0.output_descs,
              (std::vector<TensorDesc *>{&cuda_plan.tensor_descs.at(8)}));
  }

  {
    WeightManagerWrapper wmw(TestWeightManager{});
    std::vector<float> q_o_weight(static_cast<size_t>(2048) * 2048);
    for (size_t i = 0, i_end = static_cast<size_t>(2048) * 2048; i < i_end;
         ++i) {
      q_o_weight[i] = static_cast<float>(i % 100) / 10000.F;
    }
    std::vector<float> k_v_weight(static_cast<size_t>(256) * 2048);
    for (size_t i = 0, i_end = static_cast<size_t>(256) * 2048; i < i_end;
         ++i) {
      k_v_weight[i] = static_cast<float>(i % 100) / 10000.F;
    }
    wmw.set_tensor("q_weight", {.dtype = DataType::kFloat32,
                                .shape = {2048, 2048},
                                .data = q_o_weight.data(),
                                .data_len = q_o_weight.size() * sizeof(float)});
    wmw.set_tensor("o_weight", {.dtype = DataType::kFloat32,
                                .shape = {2048, 2048},
                                .data = q_o_weight.data(),
                                .data_len = q_o_weight.size() * sizeof(float)});
    wmw.set_tensor("k_weight", {.dtype = DataType::kFloat32,
                                .shape = {256, 2048},
                                .data = k_v_weight.data(),
                                .data_len = k_v_weight.size() * sizeof(float)});
    wmw.set_tensor("v_weight", {.dtype = DataType::kFloat32,
                                .shape = {256, 2048},
                                .data = k_v_weight.data(),
                                .data_len = k_v_weight.size() * sizeof(float)});

    cuda::CudaRuntime cuda_runtime(std::move(cuda_plan), wmw);
    EXPECT_TRUE(
        cuda_runtime.input_names() ==
        (std::vector<std::string>{"hidden_state", "cos", "sin", "pos_ids"}));
    EXPECT_TRUE(cuda_runtime.output_names() == std::vector<std::string>{"out"});

    {
      SafeTensorWeightManager swm(utils::BazelRunfile::RLocation(
          "tiny_llm/tests/datas/llama_attn_res_b1q5h2048.safetensors"));
      auto cos_view = swm.get_tensor("cos");
      Tensor cos({.type = DeviceType::kCpu, .id = 0}, DataType::kFloat32,
                 {5, 32}, {}, 0,
                 std::make_shared<Buffer>(
                     // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
                     const_cast<void *>(cos_view.data), cos_view.data_len,
                     Device{.type = DeviceType::kCpu, .id = 0}));
      auto sin_view = swm.get_tensor("sin");
      Tensor sin({.type = DeviceType::kCpu, .id = 0}, DataType::kFloat32,
                 {5, 32}, {}, 0,
                 std::make_shared<Buffer>(
                     // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
                     const_cast<void *>(sin_view.data), sin_view.data_len,
                     Device{.type = DeviceType::kCpu, .id = 0}));
      cuda_runtime.cpu_tensor_copy_to_input("cos", cos);
      cuda_runtime.cpu_tensor_copy_to_input("sin", sin);
      {
        Tensor hidden_state({.type = DeviceType::kCpu, .id = 0},
                            DataType::kFloat32, {1, 3, 2048}, true);
        {
          auto *hidden_state_ptr = hidden_state.data<float>();
          for (size_t i = 0, i_end = static_cast<size_t>(3) * 2048; i < i_end;
               ++i) {
            hidden_state_ptr[i] = static_cast<float>(i % 100) / 10000.F;
          }
        }
        Tensor pos_ids({.type = DeviceType::kCpu, .id = 0}, DataType::kUint32,
                       {1, 3}, true);
        {
          auto *pos_ids_ptr = pos_ids.data<uint32_t>();
          for (uint32_t i = 0; i < 3; ++i) {
            pos_ids_ptr[i] = i;
          }
        }

        cuda_runtime.cpu_tensor_copy_to_input("hidden_state", hidden_state);
        cuda_runtime.cpu_tensor_copy_to_input("pos_ids", pos_ids);
        cuda_runtime.execute();

        Tensor out({.type = DeviceType::kCpu}, DataType::kFloat32, {2}, true);
        cuda_runtime.output_copy_to_cpu_tensor("out", out);
        EXPECT_EQ(out.shape(), (std::vector<int64_t>{1, 3, 2048}));
        {
          const auto *out_ptr = out.data<float>();
          const auto *target_ptr =
              reinterpret_cast<const float *>(swm.get_tensor("res").data);
          for (uint32_t j = 0, j_end = 6144; j < j_end; ++j) {
            EXPECT_TRUE(std::abs(out_ptr[j] - target_ptr[j]) < 1e-5F);
          }
        }
      }
      cuda_runtime.set_prefill(false);
      {
        Tensor hidden_state({.type = DeviceType::kCpu, .id = 0},
                            DataType::kFloat32, {1, 1, 2048}, true);
        {
          auto *hidden_state_ptr = hidden_state.data<float>();
          for (size_t i = 0; i < 2048; ++i) {
            hidden_state_ptr[i] =
                static_cast<float>((i + static_cast<size_t>(3) * 2048) % 100) /
                10000.F;
          }
        }
        Tensor pos_ids({.type = DeviceType::kCpu, .id = 0}, DataType::kUint32,
                       {1, 1}, true);
        *pos_ids.data<uint32_t>() = 3;

        cuda_runtime.cpu_tensor_copy_to_input("hidden_state", hidden_state);
        cuda_runtime.cpu_tensor_copy_to_input("pos_ids", pos_ids);
        cuda_runtime.execute();

        Tensor out({.type = DeviceType::kCpu}, DataType::kFloat32, {2}, true);
        cuda_runtime.output_copy_to_cpu_tensor("out", out);
        EXPECT_EQ(out.shape(), (std::vector<int64_t>{1, 1, 2048}));
        {
          const auto *out_ptr = out.data<float>();
          const auto *target_ptr =
              reinterpret_cast<const float *>(swm.get_tensor("res").data) +
              (static_cast<size_t>(3) * 2048);
          for (uint32_t j = 0; j < 2048; ++j) {
            EXPECT_TRUE(std::abs(out_ptr[j] - target_ptr[j]) < 1e-5F);
          }
        }
      }
      {
        Tensor hidden_state({.type = DeviceType::kCpu, .id = 0},
                            DataType::kFloat32, {1, 1, 2048}, true);
        {
          auto *hidden_state_ptr = hidden_state.data<float>();
          for (size_t i = 0; i < 2048; ++i) {
            hidden_state_ptr[i] =
                static_cast<float>((i + static_cast<size_t>(4) * 2048) % 100) /
                10000.F;
          }
        }
        Tensor pos_ids({.type = DeviceType::kCpu, .id = 0}, DataType::kUint32,
                       {1, 1}, true);
        *pos_ids.data<uint32_t>() = 4;

        cuda_runtime.cpu_tensor_copy_to_input("hidden_state", hidden_state);
        cuda_runtime.cpu_tensor_copy_to_input("pos_ids", pos_ids);
        cuda_runtime.execute();

        Tensor out({.type = DeviceType::kCpu}, DataType::kFloat32, {2}, true);
        cuda_runtime.output_copy_to_cpu_tensor("out", out);
        EXPECT_EQ(out.shape(), (std::vector<int64_t>{1, 1, 2048}));
        {
          const auto *out_ptr = out.data<float>();
          const auto *target_ptr =
              reinterpret_cast<const float *>(swm.get_tensor("res").data) +
              (static_cast<size_t>(4) * 2048);
          for (uint32_t j = 0; j < 2048; ++j) {
            EXPECT_TRUE(std::abs(out_ptr[j] - target_ptr[j]) < 1e-5F);
          }
        }
      }
    }
  }
}
} // namespace tiny_llm
