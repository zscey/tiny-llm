#pragma once

#include "tiny_llm/tensor/tensor.hpp"
#include <cstdint>
#include <span>
#include <vector>

namespace tiny_llm {

struct PagePoolConfig {
  uint32_t num_pages{};
  uint32_t num_heads{};
  uint32_t page_size{};
  uint32_t head_dim{};

  DeviceType device_type{DeviceType::kCpu};
  DataType data_type{DataType::kFloat32};
};

class PagePool {
public:
  explicit PagePool(PagePoolConfig config);

  PagePool(const PagePool &) = delete;
  auto operator=(const PagePool &) -> PagePool & = delete;
  PagePool(PagePool &&) noexcept = default;
  auto operator=(PagePool &&) noexcept -> PagePool & = default;
  ~PagePool() = default;

  auto k_pool_ptr() -> void *;
  auto v_pool_ptr() -> void *;

  auto allocate_pages(uint32_t num_logical_pages) -> std::vector<int32_t>;
  void free_pages(std::span<const int32_t> page_ids);

  [[nodiscard]] auto free_count() const -> uint32_t;
  [[nodiscard]] auto total_pages() const -> uint32_t;

private:
  Tensor k_buffer_{{.type = DeviceType::kCpu, .id = 0}, DataType::kFloat32};
  Tensor v_buffer_{{.type = DeviceType::kCpu, .id = 0}, DataType::kFloat32};
  std::vector<int32_t> free_list_;
};

} // namespace tiny_llm
