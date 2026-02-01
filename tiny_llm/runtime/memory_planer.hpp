#pragma once

#include "tiny_llm/common/common_macros.hpp"
#include <concepts>
#include <cstddef>
#include <memory>

namespace tiny_llm {
struct VirtualBlock {
  size_t offset{};
  size_t size{};
};

template <typename T>
concept MemoryPlanner =
    requires(T t, size_t size, size_t alignment, VirtualBlock v_block) {
      { t.allocate(size, alignment) } -> std::same_as<VirtualBlock>;
      { t.deallocate(v_block) } -> std::same_as<void>;
    };

class MemoryPlannerWrapper {
  struct Concept {
    Concept() = default;
    TINY_LLM_DEFAULT_COPY_MOVE(Concept);
    virtual ~Concept() = default;

    virtual auto allocate(size_t size, size_t alignment) -> VirtualBlock = 0;
    virtual void deallocate(VirtualBlock v_block) = 0;
  };

  template <MemoryPlanner T> struct Container : public Concept {
    T memory_planner;

    Container() = default;
    TINY_LLM_DELETE_COPY_MOVE(Container);
    ~Container() override = default;

    auto allocate(size_t size, size_t alignment) -> VirtualBlock override {
      return memory_planner.allocate(size, alignment);
    }

    void deallocate(VirtualBlock v_block) override {
      memory_planner.deallocate(v_block);
    }
  };

  std::unique_ptr<Concept> wrapper_;

public:
  template <MemoryPlanner T>
  MemoryPlannerWrapper() : wrapper_(std::make_unique<Container<T>>()) {}

  TINY_LLM_DELETE_COPY_MOVE(MemoryPlannerWrapper);
  ~MemoryPlannerWrapper() = default;

  auto allocate(size_t size, size_t alignment) -> VirtualBlock {
    return wrapper_->allocate(size, alignment);
  }

  void deallocate(VirtualBlock v_block) { wrapper_->deallocate(v_block); }
};
} // namespace tiny_llm
