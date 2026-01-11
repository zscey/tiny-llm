#include "tiny_llm/weight_managers/safetensors/weight_manager.hpp"
#include "tiny_llm/common/log_and_excepts.hpp"

extern "C" {
auto safetensor_init(const char *path) -> SafeTensorContext *;
auto safetensor_get_tensor(SafeTensorContext *ctx, const char *name)
    -> tiny_llm::SliceView;
void safetensor_free(SafeTensorContext *ctx);
}

namespace tiny_llm {
SafeTensorWeightManager::SafeTensorWeightManager(const std::string &path) {
  TINY_LLM_CHECK(!path.empty());
  ctx_ = safetensor_init(path.c_str());
  TINY_LLM_CHECK(ctx_);
}

SafeTensorWeightManager::SafeTensorWeightManager(
    SafeTensorWeightManager &&other) noexcept {
  std::swap(ctx_, other.ctx_);
  std::swap(user_defined_weights_, other.user_defined_weights_);
}

auto SafeTensorWeightManager::operator=(
    SafeTensorWeightManager &&other) noexcept -> SafeTensorWeightManager & {
  if (this != std::addressof(other)) {
    std::swap(ctx_, other.ctx_);
    std::swap(user_defined_weights_, other.user_defined_weights_);
  }
  return *this;
}

auto SafeTensorWeightManager::get_tensor(const std::string &name) -> SliceView {
  TINY_LLM_CHECK(!name.empty());
  auto iter = user_defined_weights_.find(name);
  if (iter != user_defined_weights_.end()) {
    return {.data = iter->second.data(), .len = iter->second.size()};
  }

  auto res = safetensor_get_tensor(ctx_, name.c_str());
  TINY_LLM_CHECK(res.data != nullptr);
  TINY_LLM_CHECK(res.len > 0);
  return res;
}

void SafeTensorWeightManager::set_tensor(std::string name,
                                         SliceView slice_view) {
  TINY_LLM_CHECK(slice_view.len > 0);
  auto &vec = user_defined_weights_
                  .insert_or_assign(std::move(name),
                                    std::vector<uint8_t>(slice_view.len))
                  .first->second;
  std::memcpy(vec.data(), slice_view.data, slice_view.len);
}

SafeTensorWeightManager::~SafeTensorWeightManager() { safetensor_free(ctx_); }

static_assert(WeightManager<SafeTensorWeightManager>);
} // namespace tiny_llm
