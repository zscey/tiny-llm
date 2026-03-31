#include "tiny_llm/parser/llama_parser.hpp"
#include "tiny_llm/utils/runfile.hpp"
#include "gtest/gtest.h"
#include <fstream>

namespace tiny_llm {
TEST(Parser, LlamaParser) {
  std::ifstream f(
      utils::BazelRunfile::RLocation("tiny_llm/tests/datas/llama_config.json"));
  auto config = nlohmann::json::parse(f);
  f.close();
  auto graph = llama_parser(config);
  EXPECT_TRUE(is_valid(graph));
}

} // namespace tiny_llm
