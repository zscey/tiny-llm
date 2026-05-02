#pragma once

#include "tiny_llm/thread_pool/thread_pool.hpp"
#include <cstddef>
#include <cstdint>

namespace tiny_llm::cpu {
inline auto cpu_ops_thread_pool() -> ThreadPool & {
  static ThreadPool cpu_ops_thread_pool;
  return cpu_ops_thread_pool;
}

template <typename F>
  requires std::invocable<F, size_t, size_t>
auto parallel_run(F f, size_t beg, size_t end,
                  size_t task_num = cpu_ops_thread_pool().thread_num())
    -> std::conditional_t<
        std::is_same_v<std::invoke_result_t<F, size_t, size_t>, void>, void,
        std::vector<std::invoke_result_t<F, size_t, size_t>>> {
  TINY_LLM_CHECK(tiny_llm::InvalidArgumentError, task_num > 0);
  TINY_LLM_CHECK(tiny_llm::InvalidArgumentError, beg < end);
  auto interval = (end - beg + task_num - 1) / task_num;

  std::vector<std::future<std::invoke_result_t<F, size_t, size_t>>> futures;
  futures.reserve(task_num);

  auto &thread_pool = cpu_ops_thread_pool();
  while (beg + interval < end) {
    futures.emplace_back(thread_pool.enqueue(std::ref(f), beg, beg + interval));
    beg += interval;
  }
  futures.emplace_back(thread_pool.enqueue(std::ref(f), beg, end));

  if constexpr (!std::is_same_v<std::invoke_result_t<F, size_t, size_t>,
                                void>) {
    std::vector<std::invoke_result_t<F, size_t, size_t>> res;
    res.reserve(futures.size());
    for (auto &fut : futures) {
      res.emplace_back(fut.get());
    }
    return res;
  } else {
    for (auto &fut : futures) {
      fut.get();
    }
  }
}
} // namespace tiny_llm::cpu
