#include "tiny_llm/device_managers/buffer.hpp"
#include <memory>
#include <vector>

namespace tiny_llm {
enum class DataType : std::uint8_t {
  kFloat,
};

auto type_size(DataType dtype) -> size_t;

class Tensor {
public:
  Tensor(DeviceType dev_type, DataType dtype, std::vector<int64_t> shape,
         bool pre_allocate = false);

  Tensor(DeviceType dev_type, DataType dtype, std::vector<int64_t> shape,
         std::vector<int64_t> stride, size_t offset,
         std::shared_ptr<Buffer> buffer);

  [[nodiscard]] auto stride() const -> const std::vector<int64_t> & {
    return stride_;
  }

  [[nodiscard]] auto shape() const -> const std::vector<int64_t> & {
    return shape_;
  }

  [[nodiscard]] auto dtype() const -> DataType { return dtype_; }

  [[nodiscard]] auto device() const -> Device;

  auto data() -> void *;

  template <typename T> auto data() -> T * { return static_cast<T *>(data()); }

  [[nodiscard]] auto data() const -> const void *;

  template <typename T> auto data() const -> const T * {
    return static_cast<const T *>(data());
  }

private:
  DeviceType dev_type_;
  DataType dtype_;
  std::vector<int64_t> shape_;
  std::vector<int64_t> stride_;
  size_t offset_;
  std::shared_ptr<Buffer> buffer_;
};
} // namespace tiny_llm
