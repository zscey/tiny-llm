#pragma once

#include "tiny_llm/common/common_macros.hpp"
#include "tiny_llm/tensor/tensor.hpp"
#include <string>
#include <vector>

namespace tiny_llm {
class IRuntime {
public:
  IRuntime() = default;
  TINY_LLM_DEFAULT_COPY_MOVE(IRuntime);

  [[nodiscard]] virtual auto input_names() const
      -> std::vector<std::string> = 0;

  [[nodiscard]] virtual auto output_names() const
      -> std::vector<std::string> = 0;

  virtual void bind_input(const std::string &name, const Tensor &tensor) = 0;

  virtual void cpu_tensor_copy_to_input(const std::string &name,
                                        const Tensor &tensor) = 0;

  virtual void output_copy_to_cpu_tensor(const std::string &name,
                                         Tensor &tensor) const = 0;

  virtual void execute() = 0;

  virtual ~IRuntime() = default;
};
} // namespace tiny_llm
