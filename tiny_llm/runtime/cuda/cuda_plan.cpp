#include "tiny_llm/runtime/cuda/cuda_plan.hpp"
#include "tiny_llm/common/checks.hpp"
#include "tiny_llm/common/exception.hpp"
#include "tiny_llm/utils/visitor.hpp"
#include <algorithm>
#include <iterator>
#include <stack>
#include <unordered_set>

namespace tiny_llm::cuda {
namespace {
constexpr auto kKernelGenerator = Visitor{
    [](const SiLUParam &param) -> CudaKernel {
      return SiLUKernel{.param = param};
    },
    [](const EmbeddingParam &param) -> CudaKernel {
      return EmbeddingKernel{.param = param};
    },
    [](const RopeParam &param) -> CudaKernel {
      return RopeKernel{.param = param};
    },
    [](const RMSNormParam &param) -> CudaKernel {
      return RMSNormKernel{.param = param};
    },
    [](const AddParam &param) -> CudaKernel {
      return AddKernel{.param = param};
    },
    [](const MulParam &param) -> CudaKernel {
      return MulKernel{.param = param};
    },
    [](const LinearParam &param) -> CudaKernel {
      return LinearKernel{.param = param};
    },
    [](const CausalAttentionParam &param) -> CudaKernel {
      return CausalAttentionKernel{.param = param};
    },
    [](const SliceLinearParam &param) -> CudaKernel {
      return SliceLinearKernel{.param = param};
    },
};

enum class Status : std::uint8_t {
  kWhite,
  kGray,
  kBlack,
};

auto topological_order(const Graph &graph) -> std::vector<uint32_t> {
  std::vector<uint32_t> res;
  res.reserve(graph.nodes.size());
  std::vector<Status> status(graph.nodes.size(), Status::kWhite);

  std::stack<uint32_t> stack;
  for (uint32_t id = 0, id_end = graph.nodes.size(); id < id_end; ++id) {
    if (status.at(id) != Status::kWhite) {
      continue;
    }

    stack.push(id);
    while (!stack.empty()) {
      uint32_t node_id = stack.top();

      if (status.at(node_id) == Status::kWhite) {
        status[node_id] = Status::kGray;

        for (auto tensor_id : graph.nodes.at(node_id)->second.output_tensors) {
          for (auto next_node_id :
               graph.tensor_infos.at(tensor_id)->second.consumer_nodes) {
            if (status.at(next_node_id) == Status::kWhite) {
              stack.push(next_node_id);
            } else if (status.at(next_node_id) == Status::kGray) {
              TINY_LLM_THROW_ERROR(RuntimeError, "Cycle in graph.");
            }
          }
        }
      } else {
        if (status[node_id] == Status::kGray) {
          status[node_id] = Status::kBlack;
          res.emplace_back(node_id);
        }
        stack.pop();
      }
    }
  }

  std::ranges::reverse(res);
  return res;
}

template <typename From, typename To>
auto vector_convert(const std::vector<From> &from) -> std::vector<To> {
  std::vector<To> to;
  to.reserve(from.size());
  std::ranges::transform(
      from, std::back_inserter(to),
      [](const auto &elem) -> auto { return static_cast<To>(elem); });
  return to;
}

auto create_task(CudaPlan &plan, std::unordered_set<uint32_t> &cache,
                 const Graph &graph, uint32_t node_id,
                 const std::vector<uint32_t> &nodes_mapping) -> void {
  const auto &[node_name, graph_node] = graph.nodes.at(node_id).value();
  if (graph_node.input_tensors.size() != input_num(graph_node.param)) {
    TINY_LLM_THROW_ERROR(RuntimeError,
                         "The number of inputs for node [{}] is incorrect; it "
                         "should be {}, but is currently {}.",
                         node_name, input_num(graph_node.param),
                         graph_node.input_tensors.size());
  }
  if (graph_node.output_tensors.size() != output_num(graph_node.param)) {
    TINY_LLM_THROW_ERROR(RuntimeError,
                         "The number of outputs for node [{}] is incorrect; it "
                         "should be {}, but is currently {}.",
                         node_name, output_num(graph_node.param),
                         graph_node.output_tensors.size());
  }
  auto &task = plan.tasks.emplace_back(
      CudaPlan::Task{.name = node_name,
                     .kernel = std::visit(kKernelGenerator, graph_node.param)});

  task.input_descs.reserve(graph_node.input_tensors.size());
  cache.clear();
  for (auto tensor_id : graph_node.input_tensors) {
    task.input_descs.emplace_back(&plan.tensor_descs.at(tensor_id));

    const auto &producer_node =
        graph.tensor_infos.at(tensor_id)->second.producer_node;
    if (producer_node) {
      cache.emplace(*producer_node);
    }
  }
  for (auto id : cache) {
    task.predecessors.emplace_back(nodes_mapping.at(id));
  }

  task.output_descs.reserve(graph_node.output_tensors.size());
  cache.clear();
  for (auto tensor_id : graph_node.output_tensors) {
    task.output_descs.emplace_back(&plan.tensor_descs.at(tensor_id));

    const auto &consumer_nodes =
        graph.tensor_infos.at(tensor_id)->second.consumer_nodes;
    cache.insert(consumer_nodes.begin(), consumer_nodes.end());
  }
  for (auto id : cache) {
    task.successors.emplace_back(nodes_mapping.at(id));
  }
}

auto bind_descs(CudaPlan &plan, const Graph &graph, uint32_t node_id) {
  const auto &graph_node = graph.nodes.at(node_id)->second;

  auto task_id = static_cast<uint32_t>(plan.tasks.size() - 1);
  uint32_t input_id{};
  for (auto tensor_id : graph_node.input_tensors) {
    const auto &[tensor_name, graph_tensor_info] =
        graph.tensor_infos.at(tensor_id).value();
    auto &plan_tensor_desc = plan.tensor_descs.at(tensor_id);

    if (!graph_tensor_info.producer_node) {
      plan_tensor_desc.dtype = graph_tensor_info.dtype;

      if (graph.input_names.contains(tensor_name)) {
        // is input
        TINY_LLM_CHECK(
            RuntimeError,
            plan.plan_config.named_shape_ranges.contains(tensor_name));
        plan_tensor_desc.cur_shape =
            plan.plan_config.named_shape_ranges.at(tensor_name).max_shape;
        std::get<0>(plan.input_infos[tensor_name]) = tensor_id;
        std::get<1>(plan.input_infos[tensor_name])
            .emplace_back(task_id, input_id);
      } else {
        // is initializer
        TINY_LLM_CHECK(RuntimeError, std::ranges::all_of(
                                         graph_tensor_info.shape,
                                         [](auto s) -> auto { return s > 0; }));
        plan_tensor_desc.cur_shape =
            vector_convert<int64_t, size_t>(graph_tensor_info.shape);
      }

      plan_tensor_desc.max_shape = plan_tensor_desc.cur_shape;
    }

    ++input_id;
  }

  auto &cur_task = plan.tasks.back();
  std::visit(
      [input_descs = cur_task.input_descs.data(),
       output_descs = cur_task.output_descs.data()](auto &kernel) -> auto {
        kernel.dtype_shape_infer(input_descs, output_descs);
      },
      cur_task.kernel);

  uint32_t output_id{};
  for (auto tensor_id : graph_node.output_tensors) {
    const auto &tensor_name = graph.tensor_infos.at(tensor_id)->first;
    if (graph.output_names.contains(tensor_name)) {
      std::get<0>(plan.output_infos[tensor_name]) = tensor_id;
      std::get<1>(plan.output_infos[tensor_name]) =
          CudaPlan::TaskIO{.task_id = task_id, .io_id = output_id};
    }

    plan.tensor_descs.at(tensor_id).max_shape =
        plan.tensor_descs.at(tensor_id).cur_shape;

    ++output_id;
  }
}
} // namespace

auto create_cuda_plan(const Graph &graph, PlanConfig plan_config) -> CudaPlan {
  for (const auto &[_, shape_range] : plan_config.named_shape_ranges) {
    TINY_LLM_CHECK(InvalidArgumentError, shape_range.min_shape.size() ==
                                             shape_range.max_shape.size());
    for (size_t i = 0, i_end = shape_range.min_shape.size(); i < i_end; ++i) {
      TINY_LLM_CHECK(InvalidArgumentError, shape_range.min_shape.at(i) <=
                                               shape_range.max_shape.at(i));
    }
  }
  auto topo_order_nodes = topological_order(graph);
  // node id to topo id
  std::vector<uint32_t> nodes_mapping(topo_order_nodes.size());
  for (uint32_t i = 0, i_end = topo_order_nodes.size(); i < i_end; ++i) {
    nodes_mapping[topo_order_nodes[i]] = i;
  }

  CudaPlan plan;
  plan.plan_config = std::move(plan_config);
  // preserve tensor order, change graph node order to topological order
  plan.tensor_descs.reserve(graph.tensor_infos.size());
  plan.tensor_dependence.reserve(graph.tensor_infos.size());
  for (const auto &named_tensor_info : graph.tensor_infos) {
    plan.tensor_descs.emplace_back(
        TensorDesc{.name = named_tensor_info->first});
    plan.tensor_dependence.emplace_back(
        named_tensor_info->second.consumer_nodes.size());
  }
  plan.tasks.reserve(graph.nodes.size());

  std::unordered_set<uint32_t> cache;
  for (auto node_id : topo_order_nodes) {
    create_task(plan, cache, graph, node_id, nodes_mapping);
    bind_descs(plan, graph, node_id);
  }

  return plan;
}
} // namespace tiny_llm::cuda
