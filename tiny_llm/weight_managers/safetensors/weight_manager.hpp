#pragma once

#include "tiny_llm/common/common_macros.hpp"
#include "tiny_llm/weight_managers/weight_managers.hpp"
#include <unordered_map>

struct SafeTensorContext;

namespace tiny_llm {
class SafeTensorWeightManager {
public:
  explicit SafeTensorWeightManager(const std::string &path);
  SafeTensorWeightManager(const SafeTensorWeightManager &) = delete;
  auto operator=(const SafeTensorWeightManager &)
      -> SafeTensorWeightManager & = delete;
  SafeTensorWeightManager(SafeTensorWeightManager &&) noexcept;
  auto operator=(SafeTensorWeightManager &&) noexcept
      -> SafeTensorWeightManager &;
  ~SafeTensorWeightManager();

  [[nodiscard]] auto get_tensor(const std::string &name) const -> SliceView;

  void set_tensor(std::string name, SliceView slice_view);

private:
  SafeTensorContext *ctx_{};

  std::unordered_map<std::string, Tensor> user_defined_weights_;
};
} // namespace tiny_llm
