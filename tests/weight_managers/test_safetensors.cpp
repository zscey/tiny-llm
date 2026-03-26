#include "tiny_llm/utils/runfile.hpp"
#include "tiny_llm/weight_managers/safetensors/weight_manager.hpp"
#include "gtest/gtest.h"
#include <numeric>

namespace tiny_llm {
TEST(WeightManagers, SafeTensors) {
  WeightManagerWrapper manager(
      (SafeTensorWeightManager(utils::BazelRunfile::RLocation(
          "tiny_llm/tests/datas/test.safetensors"))));

  auto embedding = manager.get_tensor("embedding");
  const auto *embedding_ptr = static_cast<const float *>(embedding.data);
  EXPECT_EQ(embedding.dtype, DataType::kFloat32);
  EXPECT_EQ(embedding.shape, (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(embedding.data_len, 24);
  for (size_t i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(embedding_ptr[i], static_cast<float>(i));
  }

  auto attention = manager.get_tensor("attention");
  const auto *attention_ptr = static_cast<const float *>(attention.data);
  EXPECT_EQ(attention.dtype, DataType::kFloat32);
  EXPECT_EQ(attention.shape, (std::vector<int64_t>{2, 2}));
  EXPECT_EQ(attention.data_len, 16);
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(attention_ptr[i], static_cast<float>(10 + i));
  }

  EXPECT_ANY_THROW((void)manager.get_tensor("random"));
  std::vector<float> random_vec{100.F, 200.F};
  manager.set_tensor("random", {.dtype = DataType::kFloat32,
                                .shape = std::vector<int64_t>{2},
                                .data = random_vec.data(),
                                .data_len = 8});
  auto random = manager.get_tensor("random");
  EXPECT_EQ(random.dtype, DataType::kFloat32);
  EXPECT_EQ(random.shape, std::vector<int64_t>{2});
  EXPECT_EQ(random.data_len, 8);
  const auto *random_ptr = static_cast<const float *>(random.data);
  EXPECT_FLOAT_EQ(random_ptr[0], 100.F);
  EXPECT_FLOAT_EQ(random_ptr[1], 200.F);
  {
    std::vector<float> random_vec{50.F, 100.F};
    manager.set_tensor("random", {.dtype = DataType::kFloat32,
                                  .shape = std::vector<int64_t>{1, 2},
                                  .data = random_vec.data(),
                                  .data_len = 8});
    auto random = manager.get_tensor("random");
    EXPECT_EQ(random.dtype, DataType::kFloat32);
    EXPECT_EQ(random.shape, (std::vector<int64_t>{1, 2}));
    EXPECT_EQ(random.data_len, 8);
    const auto *random_ptr = static_cast<const float *>(random.data);
    EXPECT_FLOAT_EQ(random_ptr[0], 50.F);
    EXPECT_FLOAT_EQ(random_ptr[1], 100.F);
  }
}
} // namespace tiny_llm
