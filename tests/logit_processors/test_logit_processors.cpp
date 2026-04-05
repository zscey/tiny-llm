#include "tiny_llm/logit_processors/sort_processors.hpp"
#include "gtest/gtest.h"
#include <cstddef>
#include <unordered_set>

namespace tiny_llm {
TEST(LogitProcessors, TopK) {
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

    LogitProcessorWrapper wrapper(TopKProcessor{0});
    wrapper.apply(logit, id, valid_size);
    EXPECT_EQ(valid_size_ptr[0], 0);
    EXPECT_EQ(valid_size_ptr[1], 0);
  }
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

    LogitProcessorWrapper wrapper(TopKProcessor{1});
    wrapper.apply(logit, id, valid_size);

    EXPECT_EQ(*logit.data<float>(), 19.F);
    EXPECT_EQ(*id.data<uint32_t>(), 19);
    EXPECT_EQ(*(logit.data<float>() + 20), 39.F);
    EXPECT_EQ(*(id.data<uint32_t>() + 20), 19);
    EXPECT_EQ(valid_size_ptr[0], 1);
    EXPECT_EQ(valid_size_ptr[1], 1);
  }
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

    LogitProcessorWrapper wrapper(TopKProcessor{5});
    wrapper.apply(logit, id, valid_size);
    for (uint32_t i = 0; i < 2; ++i) {
      auto *logit_ptr = logit.data<float>() + static_cast<ptrdiff_t>(i * 20);
      auto *id_ptr = id.data<uint32_t>() + static_cast<ptrdiff_t>(i * 20);

      std::unordered_set<uint32_t> idxs;
      for (uint32_t j = 0; j < 5; ++j) {
        EXPECT_EQ(logit_ptr[j], static_cast<float>(id_ptr[j]) +
                                    (20.F * static_cast<float>(i)));
        idxs.emplace(id_ptr[j]);
      }

      EXPECT_TRUE(idxs.contains(17));
      EXPECT_TRUE(idxs.contains(18));
      EXPECT_TRUE(idxs.contains(19));
      EXPECT_TRUE(idxs.contains(15));
      EXPECT_TRUE(idxs.contains(16));

      EXPECT_EQ(valid_size.data<uint32_t>()[i], 5);
    }

    valid_size_ptr[0] = 21;
    EXPECT_ANY_THROW(wrapper.apply(logit, id, valid_size));
    valid_size_ptr[1] = 20;
    {
      LogitProcessorWrapper bad_wrapper(TopKProcessor{21});
      EXPECT_ANY_THROW(bad_wrapper.apply(logit, id, valid_size));
    }
  }
}

TEST(LogitProcessors, TopP) {
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

    LogitProcessorWrapper wrapper(TopPProcessor{1, 1});
    wrapper.apply(logit, id, valid_size);
    EXPECT_EQ(valid_size_ptr[0], 20);
    EXPECT_EQ(valid_size_ptr[1], 20);
  }
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

    LogitProcessorWrapper wrapper(TopPProcessor{0, 3});
    wrapper.apply(logit, id, valid_size);
    EXPECT_FLOAT_EQ(logit.data<float>()[0], 19.F / 40.F);
    EXPECT_FLOAT_EQ(logit.data<float>()[1], 18.F / 40.F);
    EXPECT_FLOAT_EQ(logit.data<float>()[2], 17.F / 40.F);
    EXPECT_FLOAT_EQ(id.data<uint32_t>()[0], 19);
    EXPECT_FLOAT_EQ(id.data<uint32_t>()[1], 18);
    EXPECT_FLOAT_EQ(id.data<uint32_t>()[2], 17);
    EXPECT_EQ(valid_size_ptr[0], 3);
    EXPECT_FLOAT_EQ(logit.data<float>()[20], 39.F / 40.F);
    EXPECT_FLOAT_EQ(logit.data<float>()[21], 38.F / 40.F);
    EXPECT_FLOAT_EQ(logit.data<float>()[22], 37.F / 40.F);
    EXPECT_FLOAT_EQ(id.data<uint32_t>()[20], 19);
    EXPECT_FLOAT_EQ(id.data<uint32_t>()[21], 18);
    EXPECT_FLOAT_EQ(id.data<uint32_t>()[22], 17);
    EXPECT_EQ(valid_size_ptr[1], 3);
  }
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
    logit_ptr[0] = 0.5F;
    logit_ptr[20] = 2.F;

    LogitProcessorWrapper wrapper(TopPProcessor{0.2});
    wrapper.apply(logit, id, valid_size);
    EXPECT_FLOAT_EQ(logit.data<float>()[0], 0.5F);
    EXPECT_FLOAT_EQ(logit.data<float>()[1], 19.F / 40.F);
    EXPECT_FLOAT_EQ(logit.data<float>()[2], 18.F / 40.F);
    EXPECT_FLOAT_EQ(logit.data<float>()[3], 17.F / 40.F);
    EXPECT_FLOAT_EQ(id.data<uint32_t>()[0], 0);
    EXPECT_FLOAT_EQ(id.data<uint32_t>()[1], 19);
    EXPECT_FLOAT_EQ(id.data<uint32_t>()[2], 18);
    EXPECT_FLOAT_EQ(id.data<uint32_t>()[3], 17);
    EXPECT_EQ(valid_size_ptr[0], 4);
    EXPECT_FLOAT_EQ(logit.data<float>()[20], 2.F);
    EXPECT_FLOAT_EQ(logit.data<float>()[21], 39.F / 40.F);
    EXPECT_FLOAT_EQ(id.data<uint32_t>()[20], 0);
    EXPECT_FLOAT_EQ(id.data<uint32_t>()[21], 19);
    EXPECT_EQ(valid_size_ptr[1], 2);
  }
}
} // namespace tiny_llm
