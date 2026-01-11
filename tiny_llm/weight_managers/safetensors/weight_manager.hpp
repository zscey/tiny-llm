#pragma once

#include "tiny_llm/common/common_macros.hpp"
#include "tiny_llm/weight_managers/weight_managers.hpp"
#include <unordered_map>
#include <vector>

struct SafeTensorContext;

namespace tiny_llm {
class SafeTensorWeightManager {
public:
  explicit SafeTensorWeightManager(const std::string &path);
  TINY_LLM_DELETE_COPY_MOVE(SafeTensorWeightManager);
  ~SafeTensorWeightManager();

  auto get_tensor(const std::string &name) -> SliceView;

  void set_tensor(std::string name, SliceView slice_view);

private:
  SafeTensorContext *ctx_{};

  std::unordered_map<std::string, std::vector<uint8_t>> user_defined_weights_;
};
} // namespace tiny_llm
