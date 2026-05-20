#pragma once

#include "tiny_llm/device_managers/buffer.hpp"
#include "tiny_llm/device_managers/cuda/cuda_context.hpp"
#include "tiny_llm/runtime/cuda/cuda_plan.hpp"
#include "tiny_llm/runtime/iruntime.hpp"
#include "tiny_llm/runtime/page_pool.hpp"
#include "tiny_llm/weight_managers/weight_managers.hpp"
#include <unordered_map>

namespace tiny_llm::cuda {
struct RuntimeConfig {
  uint32_t max_request_num{1};
};

struct RequestMeta {
  uint32_t seq_len{};
  uint32_t kv_len{};
  // Don't modify member `kv_pages` manually.
  std::unordered_map<std::string, std::vector<uint32_t>> kv_pages;
};

struct RuntimeMeta {
  // name -> {max_blocks, buffer}
  std::unordered_map<std::string, std::tuple<uint32_t, Tensor>> block_tables;

  Tensor seq_separator{{.type = DeviceType::kCpu}, DataType::kUint32};
  Tensor cache_offsets{{.type = DeviceType::kCpu}, DataType::kUint32};
  Tensor kv_lens{{.type = DeviceType::kCpu}, DataType::kUint32};
  uint32_t total_queries{};
  uint32_t num_requests{};
  uint32_t max_q_len{};
};

class CudaRuntime : public IRuntime {
  friend void check_input(const CudaRuntime &cuda_runtime,
                          const std::string &name, const Tensor &tensor);

public:
  CudaRuntime(CudaPlan plan, const WeightManagerWrapper &weight_manager_wrapper,
              const RuntimeConfig &config = {});

  [[nodiscard]] auto input_names() const -> std::vector<std::string> override;

  [[nodiscard]] auto output_names() const -> std::vector<std::string> override;

  void bind_input(const std::string &name, const Tensor &tensor) override;

  void cpu_tensor_copy_to_input(const std::string &name,
                                const Tensor &tensor) override;

  void output_copy_to_cpu_tensor(const std::string &name,
                                 Tensor &tensor) const override;

  void execute() override;

  void set_prefill(bool state) override;

  void bind_request_meta(std::vector<RequestMeta> &request_metas);

  void destroy_request_meta(std::vector<RequestMeta> &request_metas);

private:
  CudaPlan plan_;
  Buffer dynamic_buffer_;
  Buffer static_buffer_;

  std::unordered_map<std::string, void *> input_ptrs_;
  CudaContext ctx_;

  RuntimeConfig config_;
  bool is_prefill_{true};
  std::unordered_map<std::string, std::tuple<uint32_t>> attn_info_;
  std::unordered_map<std::string, std::tuple<uint32_t, PagePool>>
      page_attn_info_;
  // TODO(hao.lin): Expend to the concept of groups.
  RuntimeMeta runtime_meta_;
};
} // namespace tiny_llm::cuda
