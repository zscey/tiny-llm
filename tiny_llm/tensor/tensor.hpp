#pragma once

#include "tiny_llm/device_managers/buffer.hpp"
#include <memory>
#include <vector>

namespace tiny_llm {
enum class DataType : std::uint8_t {
  kFloat32,
  kUint32,
};

auto to_string(DataType dtype) -> std::string;

auto type_size(DataType dtype) -> size_t;

class Tensor {
public:
  Tensor(Device device, DataType dtype, std::vector<int64_t> shape = {},
         bool pre_allocate = false);

  Tensor(Device device, DataType dtype, std::vector<int64_t> shape,
         std::vector<int64_t> stride, size_t offset,
         std::shared_ptr<Buffer> buffer);

  Tensor(const Tensor &) = delete;
  auto operator=(const Tensor &) -> Tensor & = delete;
  Tensor(Tensor &&other) noexcept;
  auto operator=(Tensor &&other) noexcept -> Tensor &;
  ~Tensor() = default;

  [[nodiscard]] auto stride() const -> const std::vector<int64_t> & {
    return stride_;
  }

  [[nodiscard]] auto shape() const -> const std::vector<int64_t> & {
    return shape_;
  }

  [[nodiscard]] auto dtype() const -> DataType { return dtype_; }

  [[nodiscard]] auto device() const -> Device { return device_; }

  auto data() -> void *;

  template <typename T> auto data() -> T * { return static_cast<T *>(data()); }

  [[nodiscard]] auto data() const -> const void *;

  template <typename T> auto data() const -> const T * {
    return static_cast<const T *>(data());
  }

  void reallocate(std::vector<int64_t> shape);

  [[nodiscard]] auto is_continuous() const -> bool;

  // May async
  void copy_to(Tensor &tensor) const;

  // May async
  [[nodiscard]] auto to(Device device) const -> Tensor;

private:
  Device device_{};
  DataType dtype_{};
  std::vector<int64_t> shape_;
  std::vector<int64_t> stride_;
  size_t offset_{};
  std::shared_ptr<Buffer> buffer_;
};
} // namespace tiny_llm
