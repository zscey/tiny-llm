#include "tiny_llm/tensor/tensor.hpp"
#include "gtest/gtest.h"

namespace tiny_llm {
template <Device Dev, DataType Dtype> struct Types {
  static constexpr Device kDev = Dev;
  static constexpr DataType kDataType = Dtype;
};

template <typename T> class TensorBaseTest : public ::testing::Test {
public:
  TensorBaseTest() : tensor(T::kDev, T::kDataType, {3, 4, 5, 6}) {}

  // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
  Tensor tensor;
  Device device{T::kDev};
  DataType data_type{T::kDataType};
  // NOLINTEND(misc-non-private-member-variables-in-classes)
};

#ifdef TESTS_WITH_CUDA
using TestTypes = ::testing::Types<
    Types<{.type = DeviceType::kCpu, .id = 0}, DataType::kFloat32>,
    Types<{.type = DeviceType::kCudaHost, .id = 0}, DataType::kFloat32>,
    Types<{.type = DeviceType::kCuda, .id = 0}, DataType::kFloat32>>;
#else
using TestTypes = ::testing::Types<Types<DeviceType::kCpu, DataType::kFloat>>;
#endif

TYPED_TEST_SUITE(TensorBaseTest, TestTypes);

TYPED_TEST(TensorBaseTest, TensorBaseApi) {
  {
    const auto &tensor = this->tensor;
    const void *ptr{};
    EXPECT_ANY_THROW((ptr = tensor.data()));
  }
  EXPECT_NO_THROW(this->tensor.data());
  EXPECT_TRUE(this->tensor.template data<float>() != nullptr);
  EXPECT_NO_THROW(this->tensor.device().type == this->device.type);
  EXPECT_NO_THROW(this->tensor.device().id == this->device.id);
  {
    const auto &tensor = this->tensor;
    EXPECT_NO_THROW(tensor.data());
    EXPECT_TRUE(tensor.data() != nullptr);
  }

  {
    auto size = static_cast<int64_t>(type_size(this->tensor.dtype()));
    const auto &tensor =
        Tensor(this->tensor.device(), this->tensor.dtype(), {2, 3, 4, 5},
               {size * 60, size * 20, size * 5, size}, 20,
               std::make_shared<Buffer>(this->tensor.data(), 1440,
                                        this->tensor.device()));
    const void *ptr{};
    EXPECT_NO_THROW((ptr = tensor.data()));
    EXPECT_TRUE(tensor.data() != nullptr);
  }

  auto size = static_cast<int64_t>(type_size(this->tensor.dtype()));
  EXPECT_TRUE((this->tensor.shape() == std::vector<int64_t>{3, 4, 5, 6}));
  EXPECT_TRUE((this->tensor.stride() ==
               std::vector<int64_t>{size * 120, size * 30, size * 6, size}));
}
} // namespace tiny_llm
