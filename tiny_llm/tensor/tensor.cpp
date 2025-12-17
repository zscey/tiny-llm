#include "tiny_llm/tensor/tensor.hpp"
#include "tiny_llm/common/log_and_excepts.hpp"
#include "tiny_llm/device_managers/cpu/cpu_allocator.hpp"
#include <stdexcept>

#ifdef TENSOR_WITH_CUDA
#include "tiny_llm/device_managers/cuda/cuda_allocator.hpp"
#endif

namespace tiny_llm {
namespace {
auto allocate_buffer(DeviceType dev_type, size_t size, size_t alignment = 64)
    -> std::shared_ptr<Buffer> {
  switch (dev_type) {
  case DeviceType::kCpu:
    return std::make_shared<Buffer>(CpuAllocator::Allocate(size, alignment));
#ifdef TENSOR_WITH_CUDA
  case DeviceType::kCudaHost:
    return std::make_shared<Buffer>(CudaHostAllocator::Allocate(size));
  case DeviceType::kCuda:
    return std::make_shared<Buffer>(CudaAllocator::Allocate(size));
#endif
  default:
    break;
  }

  throw std::runtime_error("Unsupported device type.");
}
} // namespace
auto type_size(DataType dtype) -> size_t {
  switch (dtype) {
  case DataType::kFloat:
    return 4;
  default:
    break;
  }

  throw std::runtime_error("Unsupported data type.");
}

Tensor::Tensor(DeviceType dev_type, DataType dtype, std::vector<int64_t> shape,
               bool pre_allocate)
    : dev_type_(dev_type), dtype_(dtype), shape_(std::move(shape)), offset_(0) {
  TINY_LLM_CHECK(shape_.size() >= 1);

  stride_ = std::vector<int64_t>(shape_.size(),
                                 static_cast<int64_t>(type_size(dtype_)));
  for (auto d = static_cast<int32_t>(shape_.size()) - 2; d >= 0; --d) {
    stride_[d] = stride_[d + 1] * shape_[d + 1];
  }

  if (pre_allocate) {
    buffer_ = allocate_buffer(dev_type_, stride_[0] * shape_[0]);
  }
}

Tensor::Tensor(DeviceType dev_type, DataType dtype, std::vector<int64_t> shape,
               std::vector<int64_t> stride, size_t offset,
               std::shared_ptr<Buffer> buffer)
    : dev_type_(dev_type), dtype_(dtype), shape_(std::move(shape)),
      stride_(std::move(stride)), offset_(offset), buffer_(std::move(buffer)) {
  TINY_LLM_CHECK(buffer_ != nullptr);
  TINY_LLM_CHECK(shape_.size() == stride_.size());
  TINY_LLM_CHECK(shape_.size() >= 1);
  auto shape_cum = static_cast<int64_t>(type_size(dtype_));
  for (auto d = static_cast<int32_t>(shape_.size()) - 1; d >= 0; --d) {
    TINY_LLM_CHECK(shape_cum <= stride_[d]);
    shape_cum *= shape_[d];
  }

  TINY_LLM_CHECK(dev_type_ == buffer_->get_device().type);

  TINY_LLM_CHECK(offset + (shape_[0] * stride_[0]) <= buffer_->get_size());
}

auto Tensor::device() const -> Device {
  return {.type = dev_type_,
          .id = buffer_ ? buffer_->get_device().id : static_cast<DeviceId>(-1)};
}

auto Tensor::data() -> void * {
  if (!buffer_) {
    buffer_ = allocate_buffer(dev_type_, stride_[0] * shape_[0]);
  }
  return static_cast<uint8_t *>(buffer_->get_ptr()) + offset_;
}

auto Tensor::data() const -> const void * {
  if (!buffer_) {
    throw std::runtime_error("Tensor not allocated.");
  }
  return static_cast<const uint8_t *>(buffer_->get_ptr()) + offset_;
}
} // namespace tiny_llm
