#include "tiny_llm/tensor/tensor.hpp"
#include "tiny_llm/common/log_and_excepts.hpp"
#include "tiny_llm/device_managers/cpu/cpu_allocator.hpp"

#ifdef TENSOR_WITH_CUDA
#include "tiny_llm/device_managers/cuda/cuda_allocator.hpp"
#include "tiny_llm/device_managers/cuda/cuda_guards.hpp"
#endif

namespace tiny_llm {
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

  throw std::runtime_error(fmt::format("Unsupported device: ({}, {}).",
                                       to_string(device.type), device.id));
}
} // namespace

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

  throw std::runtime_error(
      fmt::format("Unsupported data type: {}.", to_string(dtype)));
}

Tensor::Tensor(Device device, DataType dtype, std::vector<int64_t> shape,
               bool pre_allocate)
    : device_(device), dtype_(dtype), shape_(std::move(shape)), offset_(0) {
  TINY_LLM_CHECK(shape_.size() >= 1);

  stride_ = std::vector<int64_t>(shape_.size(),
                                 static_cast<int64_t>(type_size(dtype_)));
  for (auto d = static_cast<int32_t>(shape_.size()) - 2; d >= 0; --d) {
    stride_[d] = stride_[d + 1] * shape_[d + 1];
  }

  if (pre_allocate) {
    buffer_ = allocate_buffer(device_, stride_[0] * shape_[0]);
  }
}

Tensor::Tensor(Device device, DataType dtype, std::vector<int64_t> shape,
               std::vector<int64_t> stride, size_t offset,
               std::shared_ptr<Buffer> buffer)
    : device_(device), dtype_(dtype), shape_(std::move(shape)),
      stride_(std::move(stride)), offset_(offset), buffer_(std::move(buffer)) {
  TINY_LLM_CHECK(buffer_ != nullptr);
  TINY_LLM_CHECK(shape_.size() == stride_.size());
  TINY_LLM_CHECK(shape_.size() >= 1);
  auto shape_cum = static_cast<int64_t>(type_size(dtype_));
  for (auto d = static_cast<int32_t>(shape_.size()) - 1; d >= 0; --d) {
    TINY_LLM_CHECK(shape_cum <= stride_[d]);
    shape_cum *= shape_[d];
  }

  TINY_LLM_CHECK(device_.type == buffer_->get_device().type);
  TINY_LLM_CHECK(device_.id == buffer_->get_device().id);

  TINY_LLM_CHECK(offset + (shape_[0] * stride_[0]) <= buffer_->get_size());
}

auto Tensor::data() -> void * {
  if (!buffer_) {
    buffer_ = allocate_buffer(device_, stride_[0] * shape_[0]);
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
