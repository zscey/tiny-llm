#include "tiny_llm/device_managers/cpu/cpu_allocator.hpp"
#include "tiny_llm/common/checks.hpp"
#include "tiny_llm/common/exception.hpp"

namespace tiny_llm {
auto CpuAllocator::Allocate(size_t size, size_t alignment) -> Buffer {
  TINY_LLM_CHECK(tiny_llm::InvalidArgumentError, alignment != 0);

  void *ptr{};
  if (size > 0) {
    auto alloc_size = (size + (alignment - 1)) / alignment * alignment;
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    ptr = std::aligned_alloc(alignment, alloc_size);
    TINY_LLM_CHECK(tiny_llm::RuntimeError, ptr != nullptr);
  }

  return {ptr, size, {.type = DeviceType::kCpu, .id = 0}, std::free};
}
} // namespace tiny_llm
