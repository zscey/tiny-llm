#include "tiny_llm/tensor/tensor.hpp"
#include "gtest/gtest.h"

namespace tiny_llm {
template <Device Dev, DataType Dtype> struct Types {
  static constexpr Device kDev = Dev;
  static constexpr DataType kDataType = Dtype;
};

template <typename T> class TensorBaseTest : public ::testing::Test {
public:
  TensorBaseTest()
      : empty_tensor(T::kDev, T::kDataType),
        tensor(T::kDev, T::kDataType, {3, 4, 5, 6}), device(T::kDev),
        dtype(T::kDataType) {}

  Tensor empty_tensor;
  Tensor tensor;
  Device device;
  DataType dtype;
};

#ifdef TESTS_WITH_CUDA
using TestTypes = ::testing::Types<
    Types<{.type = DeviceType::kCpu, .id = 0}, DataType::kFloat32>,
    Types<{.type = DeviceType::kCudaHost, .id = 0}, DataType::kFloat32>,
    Types<{.type = DeviceType::kCuda, .id = 0}, DataType::kFloat32>>;
#else
using TestTypes = ::testing::Types<
    Types<{.type = DeviceType::kCpu, .id = 0}, DataType::kFloat32>>;
#endif

TYPED_TEST_SUITE(TensorBaseTest, TestTypes);

TYPED_TEST(TensorBaseTest, TensorBaseApi) {
  { // empty_tensor
    EXPECT_TRUE(this->empty_tensor.dtype() == this->dtype);
    EXPECT_TRUE(this->empty_tensor.device().type == this->device.type);
    EXPECT_TRUE(this->empty_tensor.device().id == this->device.id);
    EXPECT_TRUE(this->empty_tensor.shape().empty());
    EXPECT_TRUE(this->empty_tensor.stride().empty());
    EXPECT_TRUE(this->empty_tensor.data() == nullptr);
  }

  { // non-empty tensor
    EXPECT_TRUE(this->empty_tensor.dtype() == this->dtype);
    EXPECT_TRUE(this->empty_tensor.device().type == this->device.type);
    EXPECT_TRUE(this->empty_tensor.device().id == this->device.id);
    auto size = static_cast<int64_t>(type_size(this->tensor.dtype()));
    EXPECT_TRUE((this->tensor.shape() == std::vector<int64_t>{3, 4, 5, 6}));
    EXPECT_TRUE(
        (this->tensor.stride() ==
         std::vector<int64_t>{size * 6 * 5 * 4, size * 6 * 5, size * 6, size}));

    {
      const auto &const_tensor = this->tensor;
      EXPECT_ANY_THROW((void)const_tensor.data());
    }
    EXPECT_NO_THROW(this->tensor.data());
    EXPECT_TRUE(this->tensor.template data<float>() != nullptr);
    {
      const auto &const_tensor = this->tensor;
      EXPECT_TRUE(const_tensor.data() != nullptr);
    }

    {
      auto size = static_cast<int64_t>(type_size(this->tensor.dtype()));
      const auto &tensor =
          Tensor(this->tensor.device(), this->tensor.dtype(), {2, 3, 4, 5},
                 {(((((size * 5) + 1) * 4) + 1) * 3) + 1,
                  (((size * 5) + 1) * 4) + 1, (size * 5) + 1, size},
                 20,
                 std::make_shared<Buffer>(this->tensor.data(), 1440,
                                          this->tensor.device()));
      const auto *ptr = tensor.data();
      EXPECT_TRUE(ptr != nullptr);
      EXPECT_TRUE((tensor.shape() == std::vector<int64_t>{2, 3, 4, 5}));
      EXPECT_TRUE((tensor.stride() ==
                   std::vector<int64_t>{(((((size * 5) + 1) * 4) + 1) * 3) + 1,
                                        (((size * 5) + 1) * 4) + 1,
                                        (size * 5) + 1, size}));
    }
  }
}
} // namespace tiny_llm
