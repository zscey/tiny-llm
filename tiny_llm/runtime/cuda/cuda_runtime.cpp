#include "tiny_llm/runtime/cuda/cuda_runtime.hpp"
#include "tiny_llm/common/log_and_excepts.hpp"
#include "tiny_llm/device_managers/cuda/cuda_allocator.hpp"
#include "tiny_llm/device_managers/cuda/cuda_context.hpp"
#include "tiny_llm/runtime/greedy_memory_planer.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <tuple>

namespace tiny_llm::cuda {
namespace {
constexpr size_t kAlign = 256;

auto element_num(const std::vector<size_t> &shape) -> size_t {
  return std::accumulate(
      shape.begin(), shape.end(), 1,
      [](const auto &left, const auto &right) -> auto { return left * right; });
}

struct Relation {
  uint32_t dependency{};
  std::vector<VirtualBlock> v_blocks;
};

auto is_output(const CudaPlan &cuda_plan, const CudaPlan::TaskIO &task_io)
    -> bool {
  return std::ranges::any_of(
      cuda_plan.output_infos, [&task_io](const auto &named_info) -> auto {
        return named_info.second.second.task_id == task_io.task_id &&
               named_info.second.second.io_id == task_io.io_id;
      });
}

struct SizeCalculator {
  GreedyMemoryPlaner gmp;
  std::vector<Relation> relations;
  const CudaPlan *cur_plan{};
  std::vector<size_t> offsets;
  std::unordered_map<const void *, uint32_t> desc_ptr_to_id;

  uint32_t cur_task_id{};

  void apply(const CudaPlan &plan) {
    // init
    gmp = GreedyMemoryPlaner{};
    relations.assign(plan.tasks.size(), {});
    cur_plan = &plan;
    offsets.assign(plan.tensor_descs.size(), 0);
    desc_ptr_to_id.clear();
    for (uint32_t i = 0, i_end = plan.tensor_descs.size(); i < i_end; ++i) {
      desc_ptr_to_id[&plan.tensor_descs.at(i)] = i;
    }

    // tackle inputs
    for (const auto &[_, info] : plan.input_infos) {
      const auto &desc = plan.tensor_descs.at(info.first);
      auto v_block = gmp.allocate(
          element_num(desc.max_shape) * type_size(desc.dtype), kAlign);
      offsets.at(info.first) = v_block.offset;
    }

    // tasks
    for (uint32_t i = 0, i_end = plan.tasks.size(); i < i_end; ++i) {
      cur_task_id = i;
      const auto &cur_task = plan.tasks.at(cur_task_id);
      std::visit(*this, cur_task.kernel);
      for (auto id : cur_task.predecessors) {
        auto &cur_relation = relations.at(id);
        if (cur_relation.dependency > 0) {
          --cur_relation.dependency;
          if (cur_relation.dependency == 0) {
            for (const auto &v_block : cur_relation.v_blocks) {
              gmp.deallocate(v_block);
            }
            cur_relation.v_blocks.clear();
          }
        }
      }
    }
  }

  void operator()(const SiLUKernel & /* unused*/) {
    const auto &cur_task = cur_plan->tasks.at(cur_task_id);
    auto &cur_relation = relations.at(cur_task_id);
    cur_relation.dependency = cur_task.successors.size();
    cur_relation.v_blocks.reserve(cur_task.output_descs.size());

    for (uint32_t i = 0, i_end = cur_task.output_descs.size(); i < i_end; ++i) {
      const auto *cur_desc = cur_task.output_descs.at(i);
      auto v_block = gmp.allocate(element_num(cur_desc->max_shape) *
                                      type_size(cur_desc->dtype),
                                  kAlign);
      offsets.at(desc_ptr_to_id.at(cur_desc)) = v_block.offset;
      if (!is_output(*cur_plan, {.task_id = cur_task_id, .io_id = i})) {
        cur_relation.v_blocks.emplace_back(v_block);
      }
    }
  }
};

struct WeightAssigner {
  const WeightManagerWrapper *cur_wmw{};
  CudaPlan *cur_plan{};

  size_t cur_task_id{};

  void apply(const WeightManagerWrapper &wmw, CudaPlan &plan) {
    cur_wmw = &wmw;
    cur_plan = &plan;

    for (size_t i = 0, i_end = plan.tasks.size(); i < i_end; ++i) {
      cur_task_id = i;
      std::visit(*this, plan.tasks.at(cur_task_id).kernel);
    }
  }

  void operator()(const SiLUKernel & /*unused*/) {}
};

template <typename From, typename To>
auto vector_convert(const std::vector<From> &from) -> std::vector<To> {
  std::vector<To> to;
  to.reserve(from.size());
  std::ranges::transform(
      from, std::back_inserter(to),
      [](const auto &elem) -> auto { return static_cast<To>(elem); });
  return to;
}

} // namespace

CudaRuntime::CudaRuntime(CudaPlan plan,
                         const WeightManagerWrapper &weight_manager_wrapper)
    : plan_(std::move(plan)), ctx_{CudaContextAllocator::CreateCudaContext()} {
  ThreadCudaContextsGuard guard(ctx_);

  SizeCalculator size_calculator;
  size_calculator.apply(plan_);

  buffer_ = CudaAllocator::Allocate(size_calculator.gmp.total_size());
  auto *base_ptr = static_cast<std::byte *>(buffer_.get_ptr());
  // input ptrs
  for (const auto &[name, info] : plan_.input_infos) {
    input_ptrs_[name] = base_ptr + size_calculator.offsets.at(info.first);
  }

  for (auto &task : plan_.tasks) {
    task.inputs.resize(task.input_descs.size());
    for (uint32_t i = 0, i_end = task.inputs.size(); i < i_end; ++i) {
      task.inputs.at(i) =
          base_ptr +
          size_calculator.offsets.at(
              size_calculator.desc_ptr_to_id.at(task.input_descs.at(i)));
    }

    task.outputs.resize(task.output_descs.size());
    for (uint32_t i = 0, i_end = task.outputs.size(); i < i_end; ++i) {
      task.outputs.at(i) =
          base_ptr +
          size_calculator.offsets.at(
              size_calculator.desc_ptr_to_id.at(task.output_descs.at(i)));
    }
  }

  WeightAssigner{}.apply(weight_manager_wrapper, plan_);
}

[[nodiscard]] auto CudaRuntime::input_names() const
    -> std::vector<std::string> {
  std::vector<std::string> res;
  res.reserve(input_ptrs_.size());
  std::ranges::transform(input_ptrs_, std::back_inserter(res),
                         [](const auto &elem) -> auto { return elem.first; });
  return res;
}

[[nodiscard]] auto CudaRuntime::output_names() const
    -> std::vector<std::string> {
  std::vector<std::string> res;
  res.reserve(plan_.output_infos.size());
  std::ranges::transform(plan_.output_infos, std::back_inserter(res),
                         [](const auto &elem) -> auto { return elem.first; });
  return res;
}

void check_input(const CudaRuntime &cuda_runtime, const std::string &name,
                 const Tensor &tensor) {
  TINY_LLM_CHECK(cuda_runtime.plan_.input_infos.contains(name));
  TINY_LLM_CHECK(cuda_runtime.input_ptrs_.contains(name));

  const auto &desc = cuda_runtime.plan_.tensor_descs.at(
      cuda_runtime.plan_.input_infos.at(name).first);
  TINY_LLM_CHECK(desc.dtype == tensor.dtype());
  TINY_LLM_CHECK(desc.cur_shape.size() == tensor.shape().size());
  TINY_LLM_CHECK((desc.max_shape <=>
                  vector_convert<int64_t, size_t>(tensor.shape())) >= 0);
}

void CudaRuntime::bind_input(const std::string &name, const Tensor &tensor) {
  TINY_LLM_CHECK(tensor.device().type == DeviceType::kCuda);
  TINY_LLM_CHECK(tensor.device().id == ctx_.id);
  check_input(*this, name, tensor);

  auto &[desc_id, task_ios] = plan_.input_infos.at(name);
  auto &desc = plan_.tensor_descs.at(desc_id);
  desc.cur_shape = vector_convert<int64_t, size_t>(tensor.shape());
  for (auto &task_io : task_ios) {
    plan_.tasks.at(task_io.task_id).inputs.at(task_io.io_id) = tensor.data();
  }
}

void CudaRuntime::cpu_tensor_copy_to_input(const std::string &name,
                                           const Tensor &tensor) {
  TINY_LLM_CHECK((tensor.device().type == DeviceType::kCudaHost ||
                  tensor.device().type == DeviceType::kCpu));
  check_input(*this, name, tensor);

  auto &[desc_id, task_ios] = plan_.input_infos.at(name);
  auto &desc = plan_.tensor_descs.at(desc_id);
  desc.cur_shape = vector_convert<int64_t, size_t>(tensor.shape());

  Tensor dst_tensor({.type = DeviceType::kCuda, .id = ctx_.id}, desc.dtype,
                    tensor.shape(), {}, 0,
                    std::make_shared<Buffer>(
                        input_ptrs_.at(name),
                        element_num(desc.max_shape) * type_size(desc.dtype),
                        Device{.type = DeviceType::kCuda, .id = ctx_.id}));
  tensor.copy_to(dst_tensor);
  for (const auto &task_io : task_ios) {
    plan_.tasks.at(task_io.task_id).inputs.at(task_io.io_id) =
        input_ptrs_.at(name);
  }

  ThreadCudaContexts::Synchronize();
}

void CudaRuntime::output_copy_to_cpu_tensor(const std::string &name,
                                            Tensor &tensor) const {
  TINY_LLM_CHECK((tensor.device().type == DeviceType::kCudaHost ||
                  tensor.device().type == DeviceType::kCpu));
  TINY_LLM_CHECK(plan_.output_infos.contains(name));

  const auto &[desc_id, task_io] = plan_.output_infos.at(name);
  const auto desc = plan_.tensor_descs.at(desc_id);
  TINY_LLM_CHECK(desc.dtype == tensor.dtype());

  tensor.reallocate(vector_convert<size_t, int64_t>(desc.cur_shape));
  const Tensor output_tensor(
      {.type = DeviceType::kCuda, .id = ctx_.id}, desc.dtype, tensor.shape(),
      {}, 0,
      std::make_shared<Buffer>(
          plan_.tasks.at(task_io.task_id).outputs.at(task_io.io_id),
          element_num(desc.max_shape) * type_size(desc.dtype),
          Device{.type = DeviceType::kCuda, .id = ctx_.id}));
  output_tensor.copy_to(tensor);

  ThreadCudaContexts::Synchronize();
}

void CudaRuntime::execute() {
  ThreadCudaContextsGuard guard(ctx_);

  for (auto &task : plan_.tasks) {
    std::visit(
        [&task](auto &kernel) -> auto {
          kernel.dtype_shape_infer(task.input_descs.data(),
                                   task.output_descs.data());
          kernel.execute(task.inputs.data(), task.outputs.data());
        },
        task.kernel);
  }
}
} // namespace tiny_llm::cuda
