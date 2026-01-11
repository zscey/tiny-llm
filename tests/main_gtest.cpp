#include "tiny_llm/utils/runfile.hpp"
#include "gtest/gtest.h"

auto main(int argc, char **argv) -> int {
  testing::InitGoogleTest(&argc, argv);
  tiny_llm::utils::BazelRunfile::Initialize(argv[0]);
  return RUN_ALL_TESTS();
}
