#include "tiny_llm/device_managers/cpu/cpu_allocator.hpp"
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
  TINY_LLM_CHECK(alignment != 0);

  void *ptr{};
  if (size > 0) {
    auto alloc_size = (size + (alignment - 1)) / alignment * alignment;
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    ptr = std::aligned_alloc(alignment, alloc_size);
    TINY_LLM_CHECK(ptr != nullptr);
  }

  return {ptr,
          size,
          {.type = DeviceType::kCpu, .id = 0},
          std::make_unique<CpuDeleter>()};
}
} // namespace tiny_llm
