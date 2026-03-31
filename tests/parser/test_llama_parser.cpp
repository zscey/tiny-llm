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

  auto input_names = graph.input_names;
  EXPECT_TRUE(input_names.size() == 2);
  EXPECT_TRUE(input_names.contains("token_ids"));
  EXPECT_TRUE(input_names.contains("pos_ids"));
  EXPECT_TRUE(graph.output_names ==
              std::unordered_set<std::string>{"model.norm.out"});
}

} // namespace tiny_llm
