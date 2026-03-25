#include "tiny_llm/runtime/cuda/cuda_runtime.hpp"
#include "tiny_llm/common/log_and_excepts.hpp"
#include "tiny_llm/device_managers/cuda/cuda_allocator.hpp"
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
  return shape.empty()
             ? 0
             : std::accumulate(shape.begin(), shape.end(), 1,
                               [](const auto &left, const auto &right) -> auto {
                                 return left * right;
                               });
}

struct Relation {
  uint32_t dependence{};
  VirtualBlock v_block;
};

struct SizeCalculator {
  GreedyMemoryPlaner gmp;
  const CudaPlan *cur_plan{};
  std::vector<Relation> relations;
  std::vector<size_t> offsets;
  std::unordered_map<const void *, uint32_t> desc_ptr_to_id;

  uint32_t cur_task_id{};

  void apply(const CudaPlan &plan) {
    // init
    gmp = GreedyMemoryPlaner{};
    cur_plan = &plan;
    relations.assign(plan.tensor_descs.size(), {});
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
      relations.at(info.first) =
          Relation{.dependence = static_cast<uint32_t>(info.second.size()),
                   .v_block = v_block};
      offsets.at(info.first) = v_block.offset;
    }

    // tasks
    for (uint32_t i = 0, i_end = plan.tasks.size(); i < i_end; ++i) {
      cur_task_id = i;
      const auto &cur_task = plan.tasks.at(cur_task_id);
      std::visit(*this, cur_task.kernel);
      for (const auto *desc_ptr : cur_task.input_descs) {
        auto &cur_relation = relations.at(desc_ptr_to_id.at(desc_ptr));
        if (cur_relation.dependence > 0) {
          --cur_relation.dependence;
          if (cur_relation.dependence == 0) {
            gmp.deallocate(cur_relation.v_block);
          }
        }
      }
    }
  }

  void operator()(const SiLUKernel &kernel) {
    const auto &cur_task = cur_plan->tasks.at(cur_task_id);
    auto &input_relation =
        relations.at(desc_ptr_to_id.at(cur_task.input_descs.at(0)));

    auto output_desc_id = desc_ptr_to_id.at(cur_task.output_descs.at(0));
    auto &output_relation = relations.at(output_desc_id);
    output_relation.dependence = cur_plan->tensor_dependence.at(output_desc_id);
    if (kernel.inplace) {
      input_relation.dependence = 0;
      output_relation.v_block = input_relation.v_block;
    } else {
      const auto &output_desc = cur_task.output_descs.at(0);
      output_relation.v_block = gmp.allocate(
          element_num(output_desc->max_shape) * type_size(output_desc->dtype),
          kAlign);
    }
    offsets.at(output_desc_id) = output_relation.v_block.offset;
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
  ThreadCudaContextsGuard guard(ctx_);

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
  ThreadCudaContextsGuard guard(ctx_);

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
