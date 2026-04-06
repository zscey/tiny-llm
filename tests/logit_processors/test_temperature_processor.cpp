#include "tiny_llm/logit_processors/temperature_processor.hpp"
#include "gtest/gtest.h"

namespace tiny_llm {
TEST(LogitProcessors, Temperature) {
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
      logit_ptr[i] = static_cast<float>(i);
      id_ptr[i] = i % 20;
    }

    LogitProcessorWrapper wrapper(TemperatureProcessor{0.5});
    wrapper.apply(logit, id, valid_size);
    for (uint32_t i = 0; i < 40; ++i) {
      EXPECT_FLOAT_EQ(logit_ptr[i], static_cast<float>(i << 1U));
    }
  }
}
} // namespace tiny_llm
