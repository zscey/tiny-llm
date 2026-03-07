#include "tiny_llm/utils/runfile.hpp"
#include "tiny_llm/common/log_and_excepts.hpp"
#include "tools/cpp/runfiles/runfiles.h"
#include <memory>
#include <mutex>

namespace tiny_llm::utils {
namespace {
auto runfile_env() -> bazel::tools::cpp::runfiles::Runfiles ** {
  static bazel::tools::cpp::runfiles::Runfiles *ptr{};
  return &ptr;
}
} // namespace
void BazelRunfile::Initialize(const std::string &exec_path) {
  static std::once_flag flag;
  std::call_once(flag, [&exec_path]() -> void {
    *runfile_env() = bazel::tools::cpp::runfiles::Runfiles::Create(exec_path);
    TINY_LLM_CHECK(*runfile_env());
  });
}

auto BazelRunfile::RLocation(const std::string &location) -> std::string {
  auto res = (*runfile_env())->Rlocation(location);
  TINY_LLM_CHECK(!res.empty());
  return res;
}
} // namespace tiny_llm::utils
