#include "tiny_llm/common/log_and_excepts.hpp"
#include "tiny_llm/weight_managers/weight_managers.hpp"
#include <numeric>
#include <unordered_map>

namespace tiny_llm {
class TestWeightManager {
public:
  TestWeightManager() = default;
  TestWeightManager(const TestWeightManager &) = delete;
  auto operator=(const TestWeightManager &) -> TestWeightManager & = delete;
  TestWeightManager(TestWeightManager &&) noexcept = default;
  auto operator=(TestWeightManager &&) noexcept
      -> TestWeightManager & = default;
  ~TestWeightManager() = default;

  [[nodiscard]] auto get_tensor(const std::string &name) const -> SliceView {
    TINY_LLM_CHECK(!name.empty());
    auto iter = user_defined_weights_.find(name);
    TINY_LLM_CHECK(iter != user_defined_weights_.end());

    const auto &tensor = iter->second;
    return {.dtype = tensor.dtype(),
            .shape = tensor.shape(),
            .data = tensor.data(),
            .data_len =
                type_size(tensor.dtype()) *
                (tensor.shape().empty()
                     ? 0
                     : std::accumulate(
                           tensor.shape().begin(), tensor.shape().end(), 1,
                           [](int64_t left, int64_t right) -> int64_t {
                             return left * right;
                           }))};
  }

  void set_tensor(std::string name, SliceView slice_view) {
    TINY_LLM_CHECK(slice_view.data != nullptr);
    TINY_LLM_CHECK(!slice_view.shape.empty());
    TINY_LLM_CHECK(
        type_size(slice_view.dtype) *
            std::accumulate(slice_view.shape.begin(), slice_view.shape.end(), 1,
                            [](int64_t left, int64_t right) -> int64_t {
                              return left * right;
                            }) ==
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

private:
  std::unordered_map<std::string, Tensor> user_defined_weights_;
};

static_assert(WeightManager<TestWeightManager>);
} // namespace tiny_llm
