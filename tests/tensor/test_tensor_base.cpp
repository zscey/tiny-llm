#include "tiny_llm/tensor/tensor.hpp"
#include "gtest/gtest.h"
#include <cstring>
#ifdef TINY_LLM_TESTS_WITH_CUDA
#include "tiny_llm/device_managers/cuda/cuda_context.hpp"
#endif

namespace tiny_llm {
namespace {
template <typename T> auto builtin_type_to_dtype() -> DataType;
template <> [[maybe_unused]] auto builtin_type_to_dtype<float>() -> DataType {
  return DataType::kFloat32;
}
template <>
[[maybe_unused]] auto builtin_type_to_dtype<uint32_t>() -> DataType {
  return DataType::kUint32;
}

[[maybe_unused]] void test_tensor_attrs(const Tensor &tensor, Device device,
                                        DataType dtype,
                                        const std::vector<int64_t> &shape,
                                        const std::vector<int64_t> &stride,
                                        size_t element_size,
                                        bool is_continuous) {
  EXPECT_EQ(tensor.device().type, device.type);
  EXPECT_EQ(tensor.device().id, device.id);
  EXPECT_EQ(tensor.dtype(), dtype);
  EXPECT_EQ(tensor.shape(), shape);
  EXPECT_EQ(tensor.stride(), stride);
  EXPECT_EQ(tensor.element_size(), element_size);
  EXPECT_EQ(tensor.is_continuous(), is_continuous);
}
} // namespace

template <typename T> class TensorBaseTest : public ::testing::Test {
public:
  using BuiltinType = T;
  TensorBaseTest()
      : empty_tensor({.type = DeviceType::kCpu, .id = 0},
                     builtin_type_to_dtype<T>()),
        tensor({.type = DeviceType::kCpu, .id = 0}, builtin_type_to_dtype<T>(),
               {3, 4, 5, 6}) {}

  Tensor empty_tensor;
  Tensor tensor;
};

using TestTypes = ::testing::Types<float, uint32_t>;
TYPED_TEST_SUITE(TensorBaseTest, TestTypes);

TYPED_TEST(TensorBaseTest, PlatformUnrelated) {
  Device device{.type = DeviceType::kCpu, .id = 0};
  DataType dtype = builtin_type_to_dtype<typename TestFixture::BuiltinType>();
  auto type_size_i64 =
      static_cast<int64_t>(sizeof(typename TestFixture::BuiltinType));

  { // empty case
    Tensor cur_tensor = std::move(this->empty_tensor);
    test_tensor_attrs(cur_tensor, device, dtype, {}, {}, 0, true);
    cur_tensor.reshape({});
    test_tensor_attrs(cur_tensor, device, dtype, {}, {}, 0, true);
    cur_tensor.reallocate({1});
    test_tensor_attrs(cur_tensor, device, dtype, {1}, {type_size_i64}, 1, true);
    const auto &tensor_ref = cur_tensor;
    EXPECT_NO_THROW((void)tensor_ref.data());
    EXPECT_NO_THROW(cur_tensor.data());
  }

  { // non-empty case
    Tensor cur_tensor = std::move(this->tensor);
    test_tensor_attrs(cur_tensor, device, dtype, {3, 4, 5, 6},
                      {type_size_i64 * 120, type_size_i64 * 30,
                       type_size_i64 * 6, type_size_i64},
                      360, true);
    cur_tensor.reshape({3, -1, 6});
    test_tensor_attrs(cur_tensor, device, dtype, {3, 20, 6},
                      {type_size_i64 * 120, type_size_i64 * 6, type_size_i64},
                      360, true);
    cur_tensor.reshape({3, 4, 5, 6});
    test_tensor_attrs(cur_tensor, device, dtype, {3, 4, 5, 6},
                      {type_size_i64 * 120, type_size_i64 * 30,
                       type_size_i64 * 6, type_size_i64},
                      360, true);
    EXPECT_ANY_THROW(cur_tensor.reshape({3, 4, 6, 6}));
    EXPECT_ANY_THROW(cur_tensor.reshape({3, 0, 6}));
    EXPECT_ANY_THROW(cur_tensor.reshape({-1, -1, 6}));

    auto *ori_data = cur_tensor.data();
    cur_tensor.reallocate({3, 4, 4, 5});
    EXPECT_EQ(ori_data, cur_tensor.data());
    test_tensor_attrs(cur_tensor, device, dtype, {3, 4, 4, 5},
                      {type_size_i64 * 80, type_size_i64 * 20,
                       type_size_i64 * 5, type_size_i64},
                      240, true);
    cur_tensor.reallocate({3, 4, 5, 8});
    test_tensor_attrs(cur_tensor, device, dtype, {3, 4, 5, 8},
                      {type_size_i64 * 160, type_size_i64 * 40,
                       type_size_i64 * 8, type_size_i64},
                      480, true);

    {
      Tensor new_tensor(
          device, dtype, {2, 10}, {type_size_i64 * 12, type_size_i64}, 50,
          std::make_shared<Buffer>(cur_tensor.data(), 480, device));
      test_tensor_attrs(new_tensor, device, dtype, {2, 10},
                        {type_size_i64 * 12, type_size_i64}, 20, false);
      EXPECT_EQ(new_tensor.data(), cur_tensor.data<uint8_t>() + 50);
    }
  }
}

namespace {
[[maybe_unused]] void test_tensor_equality(const Tensor &left,
                                           const Tensor &right) {
  EXPECT_EQ(left.element_size(), right.element_size());
  EXPECT_EQ(left.dtype(), right.dtype());
  EXPECT_EQ(std::memcmp(left.data(), right.data(),
                        left.element_size() * type_size(left.dtype())),
            0);
}
} // namespace

TYPED_TEST(TensorBaseTest, Copy) {
  this->empty_tensor.data();
  this->tensor.data();

  {
    auto cpu_empty = this->empty_tensor.to({.type = DeviceType::kCpu, .id = 0});
    EXPECT_EQ(cpu_empty.data(), nullptr);
#ifdef TINY_LLM_TESTS_WITH_CUDA
    auto cuda_host_empty =
        cpu_empty.to({.type = DeviceType::kCudaHost, .id = 0});
    EXPECT_EQ(cuda_host_empty.data(), nullptr);
    auto cuda_empty = cuda_host_empty.to({.type = DeviceType::kCuda, .id = 0});
    EXPECT_EQ(cuda_empty.data(), nullptr);
    cuda_host_empty = cuda_empty.to({.type = DeviceType::kCudaHost, .id = 0});
    EXPECT_EQ(cuda_host_empty.data(), nullptr);
    cpu_empty = cuda_host_empty.to({.type = DeviceType::kCpu, .id = 0});
    EXPECT_EQ(cpu_empty.data(), nullptr);
#endif
  }

  {
    auto cpu_tensor = this->tensor.to({.type = DeviceType::kCpu, .id = 0});
    test_tensor_equality(cpu_tensor, this->tensor);
#ifdef TINY_LLM_TESTS_WITH_CUDA
    auto cuda_host_tensor =
        cpu_tensor.to({.type = DeviceType::kCudaHost, .id = 0});
    test_tensor_equality(cuda_host_tensor, cpu_tensor);
    auto cuda_tensor =
        cuda_host_tensor.to({.type = DeviceType::kCuda, .id = 0});
    int32_t dev_num{};
    cudaGetDeviceCount(&dev_num);
    if (dev_num > 1) {
      EXPECT_ANY_THROW(
          (void)cuda_tensor.to({.type = DeviceType::kCuda, .id = 1}));
    } else {
      cuda_tensor = cuda_tensor.to({.type = DeviceType::kCuda, .id = 0});
    }
    cuda_host_tensor = cuda_tensor.to({.type = DeviceType::kCudaHost, .id = 0});
    ThreadCudaContexts::Synchronize();
    test_tensor_equality(cuda_host_tensor, cpu_tensor);
    cpu_tensor = cuda_host_tensor.to({.type = DeviceType::kCpu, .id = 0});
    test_tensor_equality(cpu_tensor, this->tensor);
#endif
  }
}
} // namespace tiny_llm
