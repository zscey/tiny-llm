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
    auto *logit_ptr = logit.data<float>();
    auto *id_ptr = id.data<uint32_t>();
    for (uint32_t i = 0; i < 40; ++i) {
      logit_ptr[i] = static_cast<float>(i);
      id_ptr[i] = i % 20;
    }

    LogitProcessorWrapper wrapper(TopKProcessor{0});
    EXPECT_EQ(wrapper.apply(logit, id, 20), 0);
  }
  {
    Tensor logit({.type = DeviceType::kCpu, .id = 0}, DataType::kFloat32,
                 {2, 20});
    Tensor id({.type = DeviceType::kCpu, .id = 0}, DataType::kUint32, {2, 20});
    auto *logit_ptr = logit.data<float>();
    auto *id_ptr = id.data<uint32_t>();
    for (uint32_t i = 0; i < 40; ++i) {
      logit_ptr[i] = static_cast<float>(i);
      id_ptr[i] = i % 20;
    }

    LogitProcessorWrapper wrapper(TopKProcessor{1});
    EXPECT_EQ(wrapper.apply(logit, id, 20), 1);

    EXPECT_EQ(*logit.data<float>(), 19.F);
    EXPECT_EQ(*id.data<uint32_t>(), 19);
    EXPECT_EQ(*(logit.data<float>() + 20), 39.F);
    EXPECT_EQ(*(id.data<uint32_t>() + 20), 19);
  }
  {
    Tensor logit({.type = DeviceType::kCpu, .id = 0}, DataType::kFloat32,
                 {2, 20});
    Tensor id({.type = DeviceType::kCpu, .id = 0}, DataType::kUint32, {2, 20});
    auto *logit_ptr = logit.data<float>();
    auto *id_ptr = id.data<uint32_t>();
    for (uint32_t i = 0; i < 40; ++i) {
      logit_ptr[i] = static_cast<float>(i);
      id_ptr[i] = i % 20;
    }

    LogitProcessorWrapper wrapper(TopKProcessor{5});
    EXPECT_EQ(wrapper.apply(logit, id, 20), 5);
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
    }

    EXPECT_ANY_THROW(wrapper.apply(logit, id, 21));
    {
      LogitProcessorWrapper bad_wrapper(TopKProcessor{21});
      EXPECT_ANY_THROW(bad_wrapper.apply(logit, id, 20));
    }
  }
}
} // namespace tiny_llm
