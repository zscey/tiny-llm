#include "tiny_llm/logit_processors/topk_processor.hpp"
#include "gtest/gtest.h"
#include <unordered_set>

namespace tiny_llm {
TEST(LogitProcessors, TopK) {
  std::vector<std::vector<LogitWithId>> res;
  {
    Tensor logit({.type = DeviceType::kCpu, .id = 0}, DataType::kFloat32,
                 {2, 20});
    auto *ptr = logit.data<float>();
    for (uint32_t i = 0; i < 40; ++i) {
      ptr[i] = static_cast<float>(i);
    }

    LogitProcessorWrapper wrapper(TopKProcessor{0});
    wrapper.apply(logit, res);

    EXPECT_EQ(res.size(), 2);
    for (const auto &elem : res) {
      EXPECT_TRUE(elem.empty());
    }
  }
  {
    Tensor logit({.type = DeviceType::kCpu, .id = 0}, DataType::kFloat32,
                 {2, 20});
    auto *ptr = logit.data<float>();
    for (uint32_t i = 0; i < 40; ++i) {
      ptr[i] = static_cast<float>(i);
    }

    LogitProcessorWrapper wrapper(TopKProcessor{1});
    wrapper.apply(logit, res);

    EXPECT_EQ(res.size(), 2);
    for (const auto &elem : res) {
      EXPECT_EQ(elem.size(), 1);
    }
    EXPECT_EQ(res.at(0).at(0).logit, 19.F);
    EXPECT_EQ(res.at(0).at(0).idx, 19);
    EXPECT_EQ(res.at(1).at(0).logit, 39.F);
    EXPECT_EQ(res.at(1).at(0).idx, 19);
  }
  {
    Tensor logit({.type = DeviceType::kCpu, .id = 0}, DataType::kFloat32,
                 {2, 20});
    auto *ptr = logit.data<float>();
    for (uint32_t i = 0; i < 40; ++i) {
      ptr[i] = static_cast<float>(i);
    }

    LogitProcessorWrapper wrapper(TopKProcessor{5});
    wrapper.apply(logit, res);

    EXPECT_EQ(res.size(), 2);
    for (uint32_t i = 0; i < 2; ++i) {
      const auto &cur_res = res.at(i);
      EXPECT_EQ(cur_res.size(), 5);

      std::unordered_set<uint32_t> idxs;
      for (uint32_t j = 0; j < 5; ++j) {
        const auto &elem = cur_res.at(j);
        EXPECT_EQ(elem.logit, static_cast<float>(elem.idx) +
                                  (20.F * static_cast<float>(i)));
        idxs.emplace(elem.idx);
      }

      EXPECT_TRUE(idxs.contains(17));
      EXPECT_TRUE(idxs.contains(18));
      EXPECT_TRUE(idxs.contains(19));
      EXPECT_TRUE(idxs.contains(15));
      EXPECT_TRUE(idxs.contains(16));
    }
  }
}
} // namespace tiny_llm
