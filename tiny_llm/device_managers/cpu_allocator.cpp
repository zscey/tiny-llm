#include "tiny_llm/device_managers/cpu_allocator.hpp"
#include "tiny_llm/common/log_and_excepts.hpp"

namespace tiny_llm {
namespace {
class CpuDeleter : public IDeleter {
public:
  CpuDeleter() = default;

  TINY_LLM_DELETE_COPY_MOVE(CpuDeleter);

  ~CpuDeleter() override = default;

  void cleanup(void *ptr) override {
    if (ptr != nullptr) {
      // NOLINTNEXTLINE(hicpp-no-malloc,cppcoreguidelines-owning-memory,cppcoreguidelines-no-malloc)
      std::free(ptr);
    }
  }
};
} // namespace

auto CpuAllocator::Allocate(size_t size, size_t alignment) -> Buffer {
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  auto *ptr = std::aligned_alloc(alignment, size);
  TINY_LLM_CHECK(ptr != nullptr);

  return {ptr,
          size,
          {.type = DeviceType::kCpu, .id = -1},
          std::make_unique<CpuDeleter>()};
}
} // namespace tiny_llm
