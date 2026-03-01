#ifdef TESTS_WITH_CUDA
#include "tiny_llm/device_managers/cuda/cuda_context.hpp"
#endif
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
  // move construct
  Tensor cur_tensor = std::move(this->empty_tensor);

  { // empty_tensor
    EXPECT_TRUE(cur_tensor.dtype() == this->dtype);
    EXPECT_TRUE(cur_tensor.device().type == this->device.type);
    EXPECT_TRUE(cur_tensor.device().id == this->device.id);
    EXPECT_TRUE(cur_tensor.shape().empty());
    EXPECT_TRUE(cur_tensor.stride().empty());
    EXPECT_TRUE(cur_tensor.data() == nullptr);

    EXPECT_NO_THROW(void(cur_tensor.to({DeviceType::kCpu})));
  }

  cur_tensor = std::move(this->tensor);
  { // non-empty tensor
    EXPECT_TRUE(cur_tensor.dtype() == this->dtype);
    EXPECT_TRUE(cur_tensor.device().type == this->device.type);
    EXPECT_TRUE(cur_tensor.device().id == this->device.id);
    auto size = static_cast<int64_t>(type_size(cur_tensor.dtype()));
    EXPECT_TRUE((cur_tensor.shape() == std::vector<int64_t>{3, 4, 5, 6}));
    EXPECT_TRUE(
        (cur_tensor.stride() ==
         std::vector<int64_t>{size * 6 * 5 * 4, size * 6 * 5, size * 6, size}));

    {
      const auto &const_tensor = cur_tensor;
      EXPECT_ANY_THROW((void)const_tensor.data());
    }
    EXPECT_NO_THROW(cur_tensor.data());
    EXPECT_TRUE(cur_tensor.template data<float>() != nullptr);
    {
      const auto &const_tensor = cur_tensor;
      EXPECT_TRUE(const_tensor.data() != nullptr);
    }

    {
      auto size = static_cast<int64_t>(type_size(cur_tensor.dtype()));
      const auto &tensor =
          Tensor(cur_tensor.device(), cur_tensor.dtype(), {2, 3, 4, 5},
                 {(((((size * 5) + 1) * 4) + 1) * 3) + 1,
                  (((size * 5) + 1) * 4) + 1, (size * 5) + 1, size},
                 20,
                 std::make_shared<Buffer>(cur_tensor.data(), 1440,
                                          cur_tensor.device()));
      EXPECT_TRUE(cur_tensor.is_continuous());
      EXPECT_NO_THROW((void)(cur_tensor.to({DeviceType::kCpu, 0})));
#ifdef TESTS_WITH_CUDA
      EXPECT_NO_THROW((void)(cur_tensor.to({DeviceType::kCudaHost, 0})));
#endif
      EXPECT_FALSE(tensor.is_continuous());
      EXPECT_ANY_THROW((void)(tensor.to({DeviceType::kCpu, 0})));
      EXPECT_ANY_THROW((void)(tensor.to({DeviceType::kCudaHost, 0})));
      EXPECT_TRUE(
          tensor.data() ==
          static_cast<void *>(cur_tensor.template data<uint8_t>() + 20));
      EXPECT_TRUE((tensor.shape() == std::vector<int64_t>{2, 3, 4, 5}));
      EXPECT_TRUE((tensor.stride() ==
                   std::vector<int64_t>{(((((size * 5) + 1) * 4) + 1) * 3) + 1,
                                        (((size * 5) + 1) * 4) + 1,
                                        (size * 5) + 1, size}));
    }
  }

  { // reallocate
    const auto *ptr = cur_tensor.data();
    cur_tensor.reallocate({3, 4, 7, 4});
    EXPECT_TRUE(cur_tensor.dtype() == this->dtype);
    EXPECT_TRUE(cur_tensor.device().type == this->device.type);
    EXPECT_TRUE(cur_tensor.device().id == this->device.id);
    EXPECT_TRUE(cur_tensor.data() == ptr);
    EXPECT_TRUE(cur_tensor.shape() == (std::vector<int64_t>{3, 4, 7, 4}));
    auto size = static_cast<int64_t>(type_size(cur_tensor.dtype()));
    EXPECT_TRUE(
        (cur_tensor.stride() ==
         std::vector<int64_t>{size * 4 * 7 * 4, size * 7 * 4, size * 4, size}));

    cur_tensor.reallocate({3, 4, 7, 5});
    (void)(cur_tensor.data());
    EXPECT_TRUE(cur_tensor.dtype() == this->dtype);
    EXPECT_TRUE(cur_tensor.device().type == this->device.type);
    EXPECT_TRUE(cur_tensor.device().id == this->device.id);
    EXPECT_TRUE(cur_tensor.shape() == (std::vector<int64_t>{3, 4, 7, 5}));
    EXPECT_TRUE(
        (cur_tensor.stride() ==
         std::vector<int64_t>{size * 4 * 7 * 5, size * 7 * 5, size * 5, size}));
  }

#ifdef TESTS_WITH_CUDA
  { // copy from x to cuda
    if (cur_tensor.device().type == DeviceType::kCpu ||
        cur_tensor.device().type == DeviceType::kCudaHost) {
      EXPECT_NO_THROW((void)(cur_tensor.to({DeviceType::kCuda, 0})));
    }

    if (cur_tensor.device().type == DeviceType::kCuda) {
      EXPECT_ANY_THROW((void)(cur_tensor.to({DeviceType::kCuda, 0})));
    }
  }

  ThreadCudaContexts::Synchronize();
#endif
}
} // namespace tiny_llm
