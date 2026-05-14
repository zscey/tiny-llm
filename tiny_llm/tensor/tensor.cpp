#include "tiny_llm/tensor/tensor.hpp"
#include "tiny_llm/common/checks.hpp"
#include "tiny_llm/common/exception.hpp"
#include "tiny_llm/device_managers/copy_data.hpp"
#include "tiny_llm/device_managers/cpu/cpu_allocator.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>

#ifdef TINY_LLM_TENSOR_WITH_CUDA
#include "tiny_llm/device_managers/cuda/cuda_allocator.hpp"
#include "tiny_llm/device_managers/cuda/cuda_device_guard.hpp"
#endif

namespace tiny_llm {
auto to_string(DataType dtype) -> std::string {
  switch (dtype) {
  case DataType::kFloat32:
    return "Float32";
  case DataType::kUint32:
    return "Uint32";
  default:
    return "Unknown";
  }
}

auto type_size(DataType dtype) -> size_t {
  switch (dtype) {
  case DataType::kFloat32:
  case DataType::kUint32:
    return 4;
  default:
    break;
  }

  TINY_LLM_THROW_ERROR(RuntimeError, "Unsupported data type: {}.",
                       to_string(dtype));
}

namespace {
auto allocate_buffer(Device device, size_t size, size_t alignment = 64)
    -> std::shared_ptr<Buffer> {
  switch (device.type) {
  case DeviceType::kCpu:
    return std::make_shared<Buffer>(CpuAllocator::Allocate(size, alignment));
#ifdef TINY_LLM_TENSOR_WITH_CUDA
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

  TINY_LLM_THROW_ERROR(RuntimeError, "Unsupported device: ({}, {}).",
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
  return std::ranges::all_of(shape, [](auto elem) -> auto { return elem > 0; });
}

// TODO(hao.lin): adjust the logic after supporting negative stride.
auto is_valid_shape_and_stride(const std::vector<int64_t> &shape,
                               const std::vector<int64_t> &stride,
                               DataType dtype) {
  if (!is_valid_shape(shape)) {
    return false;
  }

  auto minimum_stride = shape_to_stride(shape, dtype);
  if (minimum_stride.size() != stride.size()) {
    return false;
  }

  for (size_t i = 0, i_end = minimum_stride.size(); i < i_end; ++i) {
    if (minimum_stride.at(i) > stride.at(i)) {
      return false;
    }
  }

  return true;
}
} // namespace

Tensor::Tensor(Device device, DataType dtype, std::vector<int64_t> shape,
               bool pre_allocate)
    : device_(device), dtype_(dtype), shape_(std::move(shape)) {
  TINY_LLM_CHECK(RuntimeError, is_valid_shape(shape_));
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
  TINY_LLM_CHECK(RuntimeError, !shape_.empty());
  if (stride_.empty()) {
    stride_ = shape_to_stride(shape_, dtype_);
  }
  TINY_LLM_CHECK(RuntimeError,
                 is_valid_shape_and_stride(shape_, stride_, dtype_));
  TINY_LLM_CHECK(RuntimeError, buffer_ != nullptr);
  TINY_LLM_CHECK(RuntimeError, device_.type == buffer_->get_device().type);
  TINY_LLM_CHECK(RuntimeError, device_.id == buffer_->get_device().id);

  TINY_LLM_CHECK(RuntimeError,
                 offset + (shape_[0] * stride_[0]) <= buffer_->get_size());
}

Tensor::Tensor(Tensor &&other) noexcept {
  std::swap(device_, other.device_);
  std::swap(dtype_, other.dtype_);
  std::swap(shape_, other.shape_);
  std::swap(stride_, other.stride_);
  std::swap(offset_, other.offset_);
  std::swap(buffer_, other.buffer_);
}

auto Tensor::operator=(Tensor &&other) noexcept -> Tensor & {
  if (this != std::addressof(other)) {
    std::swap(device_, other.device_);
    std::swap(dtype_, other.dtype_);
    std::swap(shape_, other.shape_);
    std::swap(stride_, other.stride_);
    std::swap(offset_, other.offset_);
    std::swap(buffer_, other.buffer_);
  }

  return *this;
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
    TINY_LLM_THROW_ERROR(RuntimeError, "Tensor is not allocated.");
  }
  return static_cast<const uint8_t *>(buffer_->get_ptr()) + offset_;
}

namespace {
auto element_size(const std::vector<int64_t> &shape) -> size_t {
  if (shape.empty()) {
    return 0;
  }
  return std::accumulate(shape.begin(), shape.end(), 1,
                         [](auto a, auto b) -> auto { return a * b; });
}

auto remove_negative_dim(std::vector<int64_t> &shape, int64_t element_size) {
  if (shape.empty()) {
    return;
  }

  int32_t neg_idx{-1};

  for (int32_t i = 0, i_end = static_cast<int32_t>(shape.size()); i < i_end;
       ++i) {
    const auto &cur_dim = shape.at(i);
    TINY_LLM_CHECK(RuntimeError, cur_dim != 0);
    if (cur_dim < 0) {
      TINY_LLM_CHECK(RuntimeError, cur_dim == -1);
      TINY_LLM_CHECK(RuntimeError, neg_idx < 0);
      neg_idx = i;
      continue;
    }

    TINY_LLM_CHECK(RuntimeError,
                   (element_size > 0 && element_size % cur_dim == 0));
    element_size /= cur_dim;
  }

  if (neg_idx >= 0) {
    shape.at(neg_idx) = element_size;
  } else {
    TINY_LLM_CHECK(RuntimeError, element_size == 1);
  }
}
} // namespace

auto Tensor::element_size() const -> size_t {
  return ::tiny_llm::element_size(shape_);
}

auto Tensor::is_continuous() const -> bool {
  return shape_to_stride(shape_, dtype_) == stride_;
}

void Tensor::reshape(std::vector<int64_t> shape) {
  TINY_LLM_CHECK(RuntimeError, is_continuous());
  remove_negative_dim(shape, static_cast<int64_t>(element_size()));

  shape_ = std::move(shape);
  stride_ = shape_to_stride(shape_, dtype_);
}

void Tensor::reallocate(std::vector<int64_t> shape) {
  TINY_LLM_CHECK(RuntimeError, is_valid_shape(shape));
  shape_ = std::move(shape);
  stride_ = shape_to_stride(shape_, dtype_);

  if (buffer_ != nullptr &&
      buffer_->get_size() >= (element_size() * type_size(dtype_))) {
    offset_ = 0;
    return;
  }

  *this = Tensor(device_, dtype_, shape_, true);
}

void Tensor::copy_to(Tensor &tensor) const {
  if (!is_continuous()) {
    TINY_LLM_THROW_ERROR(RuntimeError, "The source tensor is discontinuous.");
  }
  if (!tensor.is_continuous()) {
    TINY_LLM_THROW_ERROR(RuntimeError, "The target tensor is discontinuous.");
  }
  TINY_LLM_CHECK(RuntimeError, dtype_ == tensor.dtype_);
  TINY_LLM_CHECK(RuntimeError, shape_ == tensor.shape_);

  auto copy_size = element_size() * type_size(dtype_);
  if (copy_size == 0) {
    return;
  }

  copy_data(data(), device_, tensor.data(), tensor.device_, copy_size);
}

auto Tensor::to(Device device) const -> Tensor {
  Tensor res(device, dtype_, shape_);
  copy_to(res);
  return res;
}

} // namespace tiny_llm
