#pragma once
#include "tiny_llm/graph/graph.hpp"
#include "tiny_llm/runtime/cuda/cuda_kernels.hpp"
#include <string>
#include <tuple>
#include <unordered_map>

namespace tiny_llm::cuda {
struct PlanConfig {
  std::unordered_map<std::string, std::vector<size_t>> named_max_shapes;
};

struct CudaPlan {
  struct Task {
    CudaKernel kernel;
    std::vector<const TensorDesc *> input_descs;
    std::vector<TensorDesc *> output_descs;
    std::vector<const void *> inputs;
    std::vector<void *> outputs;

    // for future parallelism
    std::vector<uint32_t> predecessors;
    std::vector<uint32_t> successors;
  };

  std::vector<TensorDesc> tensor_descs;
  std::vector<Task> tasks;

  // name -> {desc_id, {task_id, input_id_in_task}}
  std::unordered_map<
      std::string,
      std::tuple<uint32_t, std::vector<std::pair<uint32_t, uint32_t>>>>
      input_infos;
  // name -> desc_id
  std::unordered_map<std::string, uint32_t> output_infos;
};

/**
 * @brief Create a cuda plan object
 *
 * @param graph a valid graph
 * @param plan_config
 * @return CudaPlan
 */
auto create_cuda_plan(const Graph &graph, const PlanConfig &plan_config)
    -> CudaPlan;
} // namespace tiny_llm::cuda
