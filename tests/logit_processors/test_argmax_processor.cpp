#include "tiny_llm/logit_processors/argmax_processor.hpp"
#include "gtest/gtest.h"
#include <cstddef>
#include <unordered_set>

namespace tiny_llm {
TEST(LogitProcessors, Argmax) {
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
      logit_ptr[i] = static_cast<float>(i) / 40.F;
      id_ptr[i] = i % 20;
    }

    LogitProcessorWrapper wrapper(ArgmaxProcessor{});
    wrapper.apply(logit, id, valid_size);
    EXPECT_EQ(logit.data<float>()[0], 19.F / 40.F);
    EXPECT_EQ(id.data<uint32_t>()[0], 19);
    EXPECT_EQ(valid_size_ptr[0], 1);
    EXPECT_EQ(logit.data<float>()[20], 39.F / 40.F);
    EXPECT_EQ(id.data<uint32_t>()[20], 19);
    EXPECT_EQ(valid_size_ptr[1], 1);
  }
}
} // namespace tiny_llm
