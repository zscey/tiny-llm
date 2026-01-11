#include "tiny_llm/utils/runfile.hpp"
#include "tiny_llm/common/log_and_excepts.hpp"
#include "tools/cpp/runfiles/runfiles.h"
#include <memory>

namespace tiny_llm::utils {
namespace {
auto runfile_env() -> std::unique_ptr<bazel::tools::cpp::runfiles::Runfiles> & {
  static std::unique_ptr<bazel::tools::cpp::runfiles::Runfiles> ptr;
  return ptr;
}
} // namespace
void BazelRunfile::Initialize(const std::string &exec_path) {
  auto &ptr = runfile_env();
  ptr.reset(bazel::tools::cpp::runfiles::Runfiles::Create(exec_path));

  TINY_LLM_CHECK(ptr);
}

auto BazelRunfile::RLocation(const std::string &location) -> std::string {
  auto res = runfile_env()->Rlocation(location);
  TINY_LLM_CHECK(!res.empty());
  return res;
}
} // namespace tiny_llm::utils
