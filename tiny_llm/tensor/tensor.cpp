#include "tiny_llm/tensor/tensor.hpp"
#include "tiny_llm/common/log_and_excepts.hpp"
#include "tiny_llm/device_managers/cpu/cpu_allocator.hpp"
#include <algorithm>

#ifdef TENSOR_WITH_CUDA
#include "tiny_llm/device_managers/cuda/cuda_allocator.hpp"
#include "tiny_llm/device_managers/cuda/cuda_guards.hpp"
#endif

namespace tiny_llm {
auto to_string(DataType dtype) -> std::string {
  switch (dtype) {
  case DataType::kFloat32:
    return "Float32";
  default:
    return "Unknow";
  }
}

auto type_size(DataType dtype) -> size_t {
  switch (dtype) {
  case DataType::kFloat32:
    return 4;
  default:
    break;
  }

  TINY_LLM_THROW_ERROR(std::runtime_error, "Unsupported data type: {}.",
                       to_string(dtype));
}

namespace {
auto allocate_buffer(Device device, size_t size, size_t alignment = 64)
    -> std::shared_ptr<Buffer> {
  switch (device.type) {
  case DeviceType::kCpu:
    return std::make_shared<Buffer>(CpuAllocator::Allocate(size, alignment));
#ifdef TENSOR_WITH_CUDA
  case DeviceType::kCudaHost:
    return std::make_shared<Buffer>(CudaHostAllocator::Allocate(size));
  case DeviceType::kCuda: {
    CudaDeviceSwitchGuard guard(device.id);
    return std::make_shared<Buffer>(CudaAllocator::Allocate(size));
  }
#endif
  default:
    break;
  }

  TINY_LLM_THROW_ERROR(std::runtime_error, "Unsupported device: ({}, {}).",
                       to_string(device.type), device.id);
}

auto shape_to_stride(const std::vector<int64_t> &shape, DataType dtype)
    -> std::vector<int64_t> {
  std::vector<int64_t> stride(shape.size(),
                              static_cast<int64_t>(type_size(dtype)));
  for (auto d = static_cast<int32_t>(shape.size()) - 2; d >= 0; --d) {
    stride[d] = stride[d + 1] * shape[d + 1];
  }
  return stride;
}

auto is_valid_shape(const std::vector<int64_t> &shape) -> bool {
  return std::ranges::all_of(shape, [](auto elem) { return elem > 0; });
}
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
auto is_valid_shape_and_stride(const std::vector<int64_t> &shape,
                               const std::vector<int64_t> &stride,
                               DataType dtype) {
  if (shape.size() != stride.size() || !is_valid_shape(shape)) {
    return false;
  }
  auto shape_cum = static_cast<int64_t>(type_size(dtype));
  for (auto d = static_cast<int32_t>(shape.size()) - 1; d >= 0; --d) {
    if (shape_cum > std::abs(stride[d])) {
      return false;
    }
    shape_cum *= shape[d];
  }
  return true;
}
} // namespace

Tensor::Tensor(Device device, DataType dtype, std::vector<int64_t> shape,
               bool pre_allocate)
    : device_(device), dtype_(dtype), shape_(std::move(shape)), offset_(0) {
  TINY_LLM_CHECK(is_valid_shape(shape_));
  stride_ = shape_to_stride(shape_, dtype_);

  if (pre_allocate) {
    buffer_ =
        allocate_buffer(device_, shape_.empty() ? 0 : stride_[0] * shape_[0]);
  }
}

Tensor::Tensor(Device device, DataType dtype, std::vector<int64_t> shape,
               std::vector<int64_t> stride, size_t offset,
               std::shared_ptr<Buffer> buffer)
    : device_(device), dtype_(dtype), shape_(std::move(shape)),
      stride_(std::move(stride)), offset_(offset), buffer_(std::move(buffer)) {
  TINY_LLM_CHECK(!shape_.empty());
  TINY_LLM_CHECK(is_valid_shape_and_stride(shape_, stride_, dtype_));
  TINY_LLM_CHECK(buffer_ != nullptr);
  TINY_LLM_CHECK(device_.type == buffer_->get_device().type);
  TINY_LLM_CHECK(device_.id == buffer_->get_device().id);

  TINY_LLM_CHECK(offset + (shape_[0] * stride_[0]) <= buffer_->get_size());
}

auto Tensor::data() -> void * {
  if (!buffer_) {
    buffer_ =
        allocate_buffer(device_, shape_.empty() ? 0 : stride_[0] * shape_[0]);
  }
  return static_cast<uint8_t *>(buffer_->get_ptr()) + offset_;
}

auto Tensor::data() const -> const void * {
  if (!buffer_) {
    TINY_LLM_THROW_ERROR(std::runtime_error, "Tensor is not allocated.");
  }
  return static_cast<const uint8_t *>(buffer_->get_ptr()) + offset_;
}
} // namespace tiny_llm
