#pragma once

#include <string>

namespace tiny_llm::utils {
class BazelRunfile {
public:
  static void Initialize(const std::string &exec_path);
  static auto RLocation(const std::string &location) -> std::string;

private:
  BazelRunfile() = default;
};
} // namespace tiny_llm::utils
