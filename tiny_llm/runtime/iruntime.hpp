#pragma once

#include "tiny_llm/common/construct_macros.hpp"
#include "tiny_llm/tensor/tensor.hpp"
#include <string>
#include <unordered_map>
#include <vector>

namespace tiny_llm {
struct RequestMeta {
  uint32_t seq_len{};
  uint32_t kv_len{};
  // Don't modify member `kv_pages` manually.
  std::unordered_map<std::string, std::vector<uint32_t>> kv_pages;
};

class IRuntime {
public:
  IRuntime() = default;
  TINY_LLM_DEFAULT_COPY_MOVE(IRuntime);

  [[nodiscard]] virtual auto input_names() const
      -> std::vector<std::string> = 0;

  [[nodiscard]] virtual auto output_names() const
      -> std::vector<std::string> = 0;

  virtual void bind_input(const std::string &name, const Tensor &tensor) = 0;

  // sync
  virtual void cpu_tensor_copy_to_input(const std::string &name,
                                        const Tensor &tensor) = 0;

  // sync
  virtual void output_copy_to_cpu_tensor(const std::string &name,
                                         Tensor &tensor) const = 0;

  virtual void execute() = 0;

  virtual void set_prefill(bool state) = 0;

  virtual void bind_request_meta(std::vector<RequestMeta> &request_metas) = 0;

  virtual void
  destroy_request_meta(std::vector<RequestMeta> &request_metas) = 0;

  virtual ~IRuntime() = default;
};
} // namespace tiny_llm
