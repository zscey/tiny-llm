#include "tiny_llm/utils/runfile.hpp"
#include "tiny_llm/weight_managers/safetensors/weight_manager.hpp"
#include "gtest/gtest.h"

namespace tiny_llm {

TEST(WeightManagers, SafeTensors) {
  auto path = tiny_llm::utils::BazelRunfile::RLocation(
      "tiny_llm/tests/datas/test.safetensors");

  SafeTensorWeightManager manager(path);

  auto embedding = manager.get_tensor("embedding");
  const auto *embedding_ptr = reinterpret_cast<const float *>(embedding.data);
  EXPECT_EQ(embedding.len, 24);
  for (size_t i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(embedding_ptr[i], static_cast<float>(i));
  }

  auto attention = manager.get_tensor("attention");
  const auto *attention_ptr = reinterpret_cast<const float *>(attention.data);
  EXPECT_EQ(attention.len, 16);
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(attention_ptr[i], static_cast<float>(10 + i));
  }

  EXPECT_ANY_THROW(manager.get_tensor("random"));
  std::vector<uint8_t> random_vec{100, 200};
  manager.set_tensor("random",
                     {.data = random_vec.data(), .len = random_vec.size()});
  auto random = manager.get_tensor("random");
  EXPECT_EQ(random.len, 2);
  EXPECT_EQ(random.data[0], 100);
  EXPECT_EQ(random.data[1], 200);
  {
    std::vector<uint8_t> random_vec{50, 100};
    manager.set_tensor("random",
                       {.data = random_vec.data(), .len = random_vec.size()});
    auto random = manager.get_tensor("random");
    EXPECT_EQ(random.len, 2);
    EXPECT_EQ(random.data[0], 50);
    EXPECT_EQ(random.data[1], 100);
  }
}
} // namespace tiny_llm
