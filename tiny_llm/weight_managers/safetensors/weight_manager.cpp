#include "tiny_llm/weight_managers/safetensors/weight_manager.hpp"
#include "tiny_llm/common/log_and_excepts.hpp"
#include <numeric>

namespace tiny_llm {
struct SliceViewRaw {
  uint32_t dtype;
  size_t shape[8];
  size_t shape_dim;
  const uint8_t *data;
  size_t data_len;
};

namespace {
auto u32_to_dtype(uint32_t dtype) -> DataType {
  if (dtype == 15) {
    return DataType::kFloat32;
  }

  TINY_LLM_THROW_ERROR(std::runtime_error, "Unsupported u32 data type: {}.",
                       dtype);
}

auto to_slice_view(const SliceViewRaw &raw) -> SliceView {
  SliceView res{.dtype = u32_to_dtype(raw.dtype),
                .shape = std::vector<int64_t>(raw.shape_dim),
                .data = raw.data,
                .data_len = raw.data_len};
  std::memcpy(res.shape.data(), &raw.shape, raw.shape_dim * sizeof(size_t));
  return res;
}

auto get_element_size(const std::vector<int64_t> &shape) -> int64_t {
  return shape.empty()
             ? 0
             : std::accumulate(shape.begin(), shape.end(), 1,
                               [](int64_t left, int64_t right) -> int64_t {
                                 return left * right;
                               });
}
} // namespace
} // namespace tiny_llm

extern "C" {
auto safetensor_init(const char *path) -> SafeTensorContext *;
auto safetensor_get_tensor(SafeTensorContext *ctx, const char *name)
    -> tiny_llm::SliceViewRaw;
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
    const auto &tensor = iter->second;
    return {.dtype = tensor.dtype(),
            .shape = tensor.shape(),
            .data = tensor.data(),
            .data_len =
                get_element_size(tensor.shape()) * type_size(tensor.dtype())};
  }

  auto raw = safetensor_get_tensor(ctx_, name.c_str());
  TINY_LLM_CHECK(raw.data != nullptr);
  TINY_LLM_CHECK(raw.data_len > 0);
  return to_slice_view(raw);
}

void SafeTensorWeightManager::set_tensor(std::string name,
                                         SliceView slice_view) {
  TINY_LLM_CHECK(slice_view.data != nullptr);
  TINY_LLM_CHECK(slice_view.data_len > 0);
  TINY_LLM_CHECK(get_element_size(slice_view.shape) *
                     type_size(slice_view.dtype) ==
                 slice_view.data_len);

  auto &tensor =
      user_defined_weights_
          .insert_or_assign(std::move(name),
                            Tensor(Device{.type = DeviceType::kCpu},
                                   slice_view.dtype,
                                   std::move(slice_view.shape), true))
          .first->second;

  std::memcpy(tensor.data(), slice_view.data, slice_view.data_len);
}

SafeTensorWeightManager::~SafeTensorWeightManager() { safetensor_free(ctx_); }

static_assert(WeightManager<SafeTensorWeightManager>);
} // namespace tiny_llm
