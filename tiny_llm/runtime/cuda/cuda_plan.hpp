#pragma once
#include "tiny_llm/graph/graph.hpp"
#include "tiny_llm/runtime/cuda/cuda_kernels.hpp"
#include <string>
#include <tuple>
#include <unordered_map>

namespace tiny_llm::cuda {
struct PlanConfig {
  struct ShapeRange {
    std::vector<size_t> min_shape;
    std::vector<size_t> max_shape;
  };
  std::unordered_map<std::string, ShapeRange> named_shape_ranges;
};

struct CudaPlan {
  struct Task {
    std::string name;

    CudaKernel kernel;
    std::vector<const TensorDesc *> input_descs;
    std::vector<TensorDesc *> output_descs;
    std::vector<const void *> inputs;
    std::vector<void *> outputs;

    // for future parallelism
    std::vector<uint32_t> predecessors;
    std::vector<uint32_t> successors;
  };

  struct TaskIO {
    uint32_t task_id{};
    uint32_t io_id{};
  };

  std::vector<TensorDesc> tensor_descs;
  std::vector<uint32_t> tensor_dependence;
  std::vector<Task> tasks;

  // name -> {desc_id, task_inputs}
  std::unordered_map<std::string, std::pair<uint32_t, std::vector<TaskIO>>>
      input_infos;
  // name -> {desc_id, task_output}
  std::unordered_map<std::string, std::pair<uint32_t, TaskIO>> output_infos;

  PlanConfig plan_config;
};

/**
 * @brief Create a cuda plan object
 *
 * @param graph a valid graph
 * @param plan_config
 * @return CudaPlan
 */
auto create_cuda_plan(const Graph &graph, PlanConfig plan_config) -> CudaPlan;
} // namespace tiny_llm::cuda
