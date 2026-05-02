#pragma once

#include "tiny_llm/common/checks.hpp"
#include "tiny_llm/common/exception.hpp"
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>

namespace tiny_llm {
class ThreadPool {
  struct alignas(64) WorkerContext {
    std::deque<std::function<void()>> local_queue;
    std::mutex mtx;
    std::condition_variable cv;
  };

public:
  explicit ThreadPool(
      size_t threads = std::max(1U, std::thread::hardware_concurrency() - 1)) {
    workers_.reserve(threads);
    contexts_.reserve(threads);

    for (size_t i = 0; i < threads; ++i) {
      auto &ctx = contexts_.emplace_back(std::make_unique<WorkerContext>());

      workers_.emplace_back([&ctx](std::stop_token st) -> void {
        while (true) {
          std::function<void()> task;
          {
            std::unique_lock lock(ctx->mtx);
            ctx->cv.wait(lock, [&]() -> bool {
              return st.stop_requested() || !ctx->local_queue.empty();
            });

            if (st.stop_requested() && ctx->local_queue.empty()) {
              break;
            }

            if (ctx->local_queue.empty()) {
              continue;
            }

            task = std::move(ctx->local_queue.front());
            ctx->local_queue.pop_front();
          }
          task();
        }
      });
    }
  }

  ~ThreadPool() {
    shutting_down_.store(true, std::memory_order_release);

    for (auto &w : workers_) {
      w.request_stop();
    }

    for (auto &ctx : contexts_) {
      ctx->cv.notify_all();
    }
  }

  template <typename F, typename... Args>
    requires std::invocable<F, Args...>
  auto enqueue(F &&f, Args &&...args)
      -> std::future<std::invoke_result_t<F, Args...>> {
    if (shutting_down_.load(std::memory_order_acquire)) {
      TINY_LLM_THROW_ERROR(tiny_llm::RuntimeError,
                           "Enqueue on stopped ThreadPool.");
    }

    auto task = std::make_shared<
        std::packaged_task<std::invoke_result_t<F, Args...>()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    size_t idx =
        next_worker_.fetch_add(1, std::memory_order_relaxed) % contexts_.size();
    {
      std::lock_guard lock(contexts_.at(idx)->mtx);
      contexts_.at(idx)->local_queue.emplace_back(
          [task]() -> auto { (*task)(); });
    }
    contexts_.at(idx)->cv.notify_one();

    return task->get_future();
  }

  ThreadPool(const ThreadPool &) = delete;
  auto operator=(const ThreadPool &) -> ThreadPool & = delete;
  ThreadPool(ThreadPool &&) = delete;
  auto operator=(ThreadPool &&) -> ThreadPool & = delete;

  auto thread_num() -> size_t { return workers_.size(); }

private:
  std::vector<std::unique_ptr<WorkerContext>> contexts_;
  std::atomic<size_t> next_worker_{0};
  std::atomic<bool> shutting_down_{false};

  // declare `workers_` at the end to support drain policy
  std::vector<std::jthread> workers_;
};
} // namespace tiny_llm
