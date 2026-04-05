#include "tiny_llm/logit_processors/sample_processors.hpp"
#include "gtest/gtest.h"
#include <cstddef>
#include <limits>
#include <unordered_set>

namespace tiny_llm {
TEST(LogitProcessors, Multinomial) {
  {
    Tensor logit({.type = DeviceType::kCpu, .id = 0}, DataType::kFloat32,
                 {2, 20});
    Tensor id({.type = DeviceType::kCpu, .id = 0}, DataType::kUint32, {2, 20});
    Tensor valid_size({.type = DeviceType::kCpu, .id = 0}, DataType::kUint32,
                      {2});
    auto *logit_ptr = logit.data<float>();
    auto *id_ptr = id.data<uint32_t>();
    auto *valid_size_ptr = valid_size.data<uint32_t>();
    valid_size_ptr[0] = 20;
    valid_size_ptr[1] = 20;
    for (uint32_t i = 0; i < 40; ++i) {
      logit_ptr[i] = std::numeric_limits<float>::lowest();
      id_ptr[i] = i % 20;
    }
    logit_ptr[1] = 1.F;
    logit_ptr[22] = 2.F;

    LogitProcessorWrapper wrapper(MultinomialProcessor{});
    wrapper.apply(logit, id, valid_size);
    EXPECT_EQ(logit.data<float>()[0], 1.F);
    EXPECT_EQ(id.data<uint32_t>()[0], 1);
    EXPECT_EQ(valid_size_ptr[0], 1);
    EXPECT_EQ(logit.data<float>()[20], 1.F);
    EXPECT_EQ(id.data<uint32_t>()[20], 2);
    EXPECT_EQ(valid_size_ptr[1], 1);
  }
}
} // namespace tiny_llm
