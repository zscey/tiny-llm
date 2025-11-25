#include "tiny_llm/device_managers/cpu_allocator.hpp"
#include "tiny_llm/common/log_and_excepts.hpp"

namespace tiny_llm {
auto CpuAllocator::Allocate(size_t size, size_t alignment) -> Buffer {
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  auto *ptr = std::aligned_alloc(alignment, size);
  TINY_LLM_CHECK(ptr != nullptr);

  return {ptr,
          size,
          {.type = DeviceType::kCpu, .id = -1},
          CpuAllocator::DefaultDeleter};
}

void CpuAllocator::DefaultDeleter(void *ptr) {
  if (ptr != nullptr) {
    // NOLINTNEXTLINE(hicpp-no-malloc,cppcoreguidelines-owning-memory,cppcoreguidelines-no-malloc)
    std::free(ptr);
  }
}
} // namespace tiny_llm
