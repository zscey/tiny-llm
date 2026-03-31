#pragma once

#include "nlohmann/json.hpp"
#include "tiny_llm/graph/graph.hpp"

namespace tiny_llm {
auto llama_parser(const nlohmann::json &config) -> Graph;
}
