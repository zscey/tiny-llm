#pragma once

#include "tiny_llm/device_managers/buffer.hpp"
#include "tiny_llm/device_managers/cuda/cuda_context.hpp"
#include "tiny_llm/runtime/cuda/cuda_plan.hpp"
#include "tiny_llm/runtime/iruntime.hpp"
#include "tiny_llm/weight_managers/weight_managers.hpp"
#include <unordered_map>

namespace tiny_llm::cuda {
class CudaRuntime : public IRuntime {
  friend void check_input(const CudaRuntime &cuda_runtime,
                          const std::string &name, const Tensor &tensor);

public:
  CudaRuntime(CudaPlan plan,
              const WeightManagerWrapper &weight_manager_wrapper);

  [[nodiscard]] auto input_names() const -> std::vector<std::string> override;

  [[nodiscard]] auto output_names() const -> std::vector<std::string> override;

  void bind_input(const std::string &name, const Tensor &tensor) override;

  // sync
  void cpu_tensor_copy_to_input(const std::string &name,
                                const Tensor &tensor) override;

  // sync
  void output_copy_to_cpu_tensor(const std::string &name,
                                 Tensor &tensor) const override;

  void execute() override;

  void set_prefill(bool state) override;

private:
  CudaPlan plan_;
  Buffer dynamic_buffer_;
  Buffer static_buffer_;

  std::unordered_map<std::string, void *> input_ptrs_;
  CudaContext ctx_;
};
} // namespace tiny_llm::cuda
