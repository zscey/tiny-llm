#include "tiny_llm/runtime/cuda/cuda_runtime.hpp"
#include "tiny_llm/common/log_and_excepts.hpp"
#include "tiny_llm/device_managers/cuda/cuda_allocator.hpp"
#include "tiny_llm/runtime/greedy_memory_planer.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <tuple>
#include <variant>

namespace tiny_llm::cuda {
namespace {
constexpr size_t kAlign = 256;

template <typename From, typename To>
auto vector_convert(const std::vector<From> &from) -> std::vector<To> {
  std::vector<To> to;
  to.reserve(from.size());
  std::ranges::transform(
      from, std::back_inserter(to),
      [](const auto &elem) -> auto { return static_cast<To>(elem); });
  return to;
}

auto element_num(const std::vector<size_t> &shape) -> size_t {
  return shape.empty()
             ? 0
             : std::accumulate(shape.begin(), shape.end(), 1,
                               [](const auto &left, const auto &right) -> auto {
                                 return left * right;
                               });
}

auto desc_to_tensor(Device device, const TensorDesc &desc, const void *data_ptr)
    -> Tensor {
  return {device,
          desc.dtype,
          vector_convert<size_t, int64_t>(desc.cur_shape),
          {},
          0,
          std::make_shared<Buffer>(
              // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
              const_cast<void *>(data_ptr),
              element_num(desc.max_shape) * type_size(desc.dtype), device)};
}

auto slice_view_to_tensor(const SliceView &slice_view) -> Tensor {
  Device device{.type = DeviceType::kCpu, .id = 0};
  return {device,
          slice_view.dtype,
          slice_view.shape,
          {},
          0,
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          std::make_shared<Buffer>(const_cast<void *>(slice_view.data),
                                   slice_view.data_len, device)};
}

struct Relation {
  uint32_t dependence{};
  VirtualBlock v_block;
};

struct SizeCalculator {
  GreedyMemoryPlaner gmp;
  const CudaPlan *cur_plan{};
  std::vector<Relation> relations;
  std::unordered_map<const CausalAttentionKernel *, std::vector<size_t>>
      attn_offsets;
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
      for (const auto *desc_ptr : cur_task.output_descs) {
        auto desc_id = desc_ptr_to_id.at(desc_ptr);
        relations.at(desc_id).dependence =
            cur_plan->tensor_dependence.at(desc_id);
      }
      // visitor
      // 1. set the `v_block` of the output
      // 2. [option, inplace] reset the `dependence` of the input to 0
      // 3. [option, initializer] set the `dependence`, `v_block`, `offset` of
      // the initializer
      // 4. [option, rope] reset the `dependence` of the outputs to 0 in the pin
      // mode
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
      for (const auto *desc_ptr : cur_task.output_descs) {
        auto desc_id = desc_ptr_to_id.at(desc_ptr);
        offsets.at(desc_id) = relations.at(desc_id).v_block.offset;
      }
    }
  }

  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  void set_v_block(const TensorDesc *from_desc, const TensorDesc *to_desc) {
    auto &to_relation = relations.at(desc_ptr_to_id.at(to_desc));
    if (from_desc != nullptr) {
      auto &from_relation = relations.at(desc_ptr_to_id.at(from_desc));
      from_relation.dependence = 0;
      to_relation.v_block = from_relation.v_block;
    } else {
      to_relation.v_block = gmp.allocate(
          element_num(to_desc->max_shape) * type_size(to_desc->dtype), kAlign);
    }
  }

  void assign_initializer(const TensorDesc *desc) {
    auto desc_id = desc_ptr_to_id.at(desc);
    auto &cur_relation = relations.at(desc_id);
    cur_relation.dependence = 0;
    cur_relation.v_block = gmp.allocate(
        element_num(desc->max_shape) * type_size(desc->dtype), kAlign);
    offsets.at(desc_id) = cur_relation.v_block.offset;
  }

  void operator()(const SiLUKernel &kernel) {
    const auto &cur_task = cur_plan->tasks.at(cur_task_id);
    set_v_block(kernel.inplace ? cur_task.input_descs.at(0) : nullptr,
                cur_task.output_descs.at(0));
  }

  void operator()(const EmbeddingKernel & /*unused*/) {
    const auto &cur_task = cur_plan->tasks.at(cur_task_id);
    assign_initializer(cur_task.input_descs.at(1));
    set_v_block(nullptr, cur_task.output_descs.at(0));
  }

  void operator()(const RopeKernel &kernel) {
    const auto &cur_task = cur_plan->tasks.at(cur_task_id);

    for (uint32_t i = 0; i < 2; ++i) {
      const auto &cur_desc = cur_task.output_descs.at(i);
      auto &cur_relation = relations.at(desc_ptr_to_id.at(cur_desc));
      if (kernel.pin) {
        cur_relation.dependence = 0;
      }
      cur_relation.v_block = gmp.allocate(element_num(cur_desc->max_shape) *
                                              type_size(cur_desc->dtype),
                                          kAlign);
    }
  }

  void operator()(const RMSNormKernel &kernel) {
    const auto &cur_task = cur_plan->tasks.at(cur_task_id);
    assign_initializer(cur_task.input_descs.at(1));
    set_v_block(kernel.inplace ? cur_task.input_descs.at(0) : nullptr,
                cur_task.output_descs.at(0));
  }

  void operator()(const AddKernel &kernel) {
    const auto &cur_task = cur_plan->tasks.at(cur_task_id);
    const TensorDesc *from_desc{};
    if (kernel.out_idx == kernel.left_idx) {
      from_desc = cur_task.input_descs.at(0);
    } else if (kernel.out_idx == kernel.right_idx) {
      from_desc = cur_task.input_descs.at(1);
    }
    set_v_block(from_desc, cur_task.output_descs.at(0));
  }

  void operator()(const MulKernel &kernel) {
    const auto &cur_task = cur_plan->tasks.at(cur_task_id);
    const TensorDesc *from_desc{};
    if (kernel.out_idx == kernel.left_idx) {
      from_desc = cur_task.input_descs.at(0);
    } else if (kernel.out_idx == kernel.right_idx) {
      from_desc = cur_task.input_descs.at(1);
    }
    set_v_block(from_desc, cur_task.output_descs.at(0));
  }

  void operator()(const LinearKernel &kernel) {
    const auto &cur_task = cur_plan->tasks.at(cur_task_id);
    assign_initializer(cur_task.input_descs.at(1));
    if (kernel.bias) {
      assign_initializer(cur_task.input_descs.at(2));
    }
    set_v_block(nullptr, cur_task.output_descs.at(0));
  }

  void operator()(const CausalAttentionKernel &kernel) {
    const auto &cur_task = cur_plan->tasks.at(cur_task_id);
    assign_initializer(cur_task.input_descs.at(4));
    assign_initializer(cur_task.input_descs.at(5));
    assign_initializer(cur_task.input_descs.at(6));
    assign_initializer(cur_task.input_descs.at(7));
    if (kernel.bias) {
      assign_initializer(cur_task.input_descs.at(8));
      assign_initializer(cur_task.input_descs.at(9));
      assign_initializer(cur_task.input_descs.at(10));
      assign_initializer(cur_task.input_descs.at(11));
    }
    set_v_block(nullptr, cur_task.output_descs.at(0));

    auto &inner_offsets = attn_offsets[&kernel];
    const auto &hidden_state_desc = cur_task.input_descs.at(0);
    auto q_o_size = element_num(hidden_state_desc->max_shape) *
                    type_size(hidden_state_desc->dtype);
    auto k_v_size = q_o_size / kernel.q_head * kernel.kv_head;
    auto q_block = gmp.allocate(q_o_size, kAlign);
    inner_offsets.emplace_back(q_block.offset);
    auto k_block = gmp.allocate(k_v_size, kAlign);
    inner_offsets.emplace_back(k_block.offset);
    auto v_block = gmp.allocate(k_v_size, kAlign);
    inner_offsets.emplace_back(v_block.offset);
    auto o_block = gmp.allocate(q_o_size, kAlign);
    inner_offsets.emplace_back(o_block.offset);
    gmp.deallocate(q_block);
    gmp.deallocate(o_block);
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

  void check_and_copy_weight(const TensorDesc *desc, const void *data_ptr,
                             const std::vector<size_t> &target_shape) const {
    TINY_LLM_CHECK(desc->cur_shape == target_shape);
    auto dst_tensor = desc_to_tensor(
        {.type = DeviceType::kCuda, .id = ThreadCudaContexts::GetContext().id},
        *desc, data_ptr);
    const auto src_tensor =
        slice_view_to_tensor(cur_wmw->get_tensor(desc->name));
    src_tensor.copy_to(dst_tensor);
  }

  void operator()(const SiLUKernel & /*unused*/) const {}

  void operator()(const EmbeddingKernel &kernel) const {
    const auto &cur_task = cur_plan->tasks.at(cur_task_id);
    check_and_copy_weight(cur_task.input_descs.at(1), cur_task.inputs.at(1),
                          {kernel.num_embeddings, kernel.hidden_size});
  }

  void operator()(const RopeKernel & /*unused*/) const {}

  void operator()(const RMSNormKernel &kernel) const {
    const auto &cur_task = cur_plan->tasks.at(cur_task_id);
    check_and_copy_weight(cur_task.input_descs.at(1), cur_task.inputs.at(1),
                          {kernel.hidden_size});
  }

  void operator()(const AddKernel & /*unused*/) const {}

  void operator()(const MulKernel & /*unused*/) const {}

  void operator()(const LinearKernel &kernel) const {
    const auto &cur_task = cur_plan->tasks.at(cur_task_id);
    check_and_copy_weight(cur_task.input_descs.at(1), cur_task.inputs.at(1),
                          {kernel.out_dim, kernel.in_dim});
    if (kernel.bias) {
      check_and_copy_weight(cur_task.input_descs.at(2), cur_task.inputs.at(2),
                            {kernel.out_dim});
    }
  }

  void operator()(const CausalAttentionKernel &kernel) const {
    auto q_dim = kernel.head_dim * kernel.q_head;
    auto kv_dim = kernel.head_dim * kernel.kv_head;

    const auto &cur_task = cur_plan->tasks.at(cur_task_id);
    check_and_copy_weight(cur_task.input_descs.at(4), cur_task.inputs.at(4),
                          {q_dim, q_dim});
    check_and_copy_weight(cur_task.input_descs.at(5), cur_task.inputs.at(5),
                          {kv_dim, q_dim});
    check_and_copy_weight(cur_task.input_descs.at(6), cur_task.inputs.at(6),
                          {kv_dim, q_dim});
    check_and_copy_weight(cur_task.input_descs.at(7), cur_task.inputs.at(7),
                          {q_dim, q_dim});
    if (kernel.bias) {
      check_and_copy_weight(cur_task.input_descs.at(8), cur_task.inputs.at(8),
                            {q_dim});
      check_and_copy_weight(cur_task.input_descs.at(9), cur_task.inputs.at(9),
                            {kv_dim});
      check_and_copy_weight(cur_task.input_descs.at(10), cur_task.inputs.at(10),
                            {kv_dim});
      check_and_copy_weight(cur_task.input_descs.at(11), cur_task.inputs.at(11),
                            {q_dim});
    }
  }
};
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

    if (std::holds_alternative<CausalAttentionKernel>(task.kernel)) {
      auto &cur_kernel = std::get<CausalAttentionKernel>(task.kernel);
      const auto &offsets = size_calculator.attn_offsets.at(&cur_kernel);
      cur_kernel.q_cache = reinterpret_cast<float *>(base_ptr + offsets.at(0));
      cur_kernel.k_cache = reinterpret_cast<float *>(base_ptr + offsets.at(1));
      cur_kernel.v_cache = reinterpret_cast<float *>(base_ptr + offsets.at(2));
      cur_kernel.o_cache = reinterpret_cast<float *>(base_ptr + offsets.at(3));
    }
  }

  WeightAssigner{}.apply(weight_manager_wrapper, plan_);
  ThreadCudaContexts::Synchronize();
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
  TINY_LLM_CHECK(
      cuda_runtime.plan_.plan_config.named_shape_ranges.contains(name));
  TINY_LLM_CHECK(cuda_runtime.input_ptrs_.contains(name));

  const auto &desc = cuda_runtime.plan_.tensor_descs.at(
      cuda_runtime.plan_.input_infos.at(name).first);
  TINY_LLM_CHECK(desc.dtype == tensor.dtype());
  TINY_LLM_CHECK(desc.cur_shape.size() == tensor.shape().size());
  const auto &min_shape =
      cuda_runtime.plan_.plan_config.named_shape_ranges.at(name).min_shape;
  for (size_t i = 0, i_end = desc.cur_shape.size(); i < i_end; ++i) {
    TINY_LLM_CHECK(static_cast<size_t>(tensor.shape().at(i)) <=
                   desc.max_shape.at(i));
    TINY_LLM_CHECK(static_cast<size_t>(tensor.shape().at(i)) >=
                   min_shape.at(i));
  }
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
  auto dst_tensor = desc_to_tensor(
      {.type = DeviceType::kCuda, .id = ThreadCudaContexts::GetContext().id},
      desc, input_ptrs_.at(name));
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
  auto output_tensor = desc_to_tensor(
      {.type = DeviceType::kCuda, .id = ThreadCudaContexts::GetContext().id},
      desc, plan_.tasks.at(task_io.task_id).outputs.at(task_io.io_id));
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

void CudaRuntime::set_prefill(bool state) {
  for (auto &task : plan_.tasks) {
    if (std::holds_alternative<CausalAttentionKernel>(task.kernel)) {
      std::get<CausalAttentionKernel>(task.kernel).is_prefill = state;
    }
  }
}
} // namespace tiny_llm::cuda
