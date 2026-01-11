#pragma once

#include "tiny_llm/common/common_macros.hpp"
#include "tiny_llm/graph/graph.hpp"
#include "tiny_llm/weight_managers/weight_managers.hpp"
#include <concepts>
#include <memory>

namespace tiny_llm {
template <typename T>
concept GraphPass = std::move_constructible<T> &&
                    requires(T t, Graph &g, WeightManagerWrapper &w) {
                      { t.run(g, w) } -> std::same_as<void>;
                    };

class PassManager {
  struct PassConcept {
    PassConcept() = default;
    TINY_LLM_DEFAULT_COPY_MOVE(PassConcept);
    virtual ~PassConcept() = default;

    virtual void run(Graph &g, WeightManagerWrapper &w) = 0;
  };

  template <GraphPass T> struct Pass final : public PassConcept {
    explicit Pass(T &&pass) : pass_concept(std::move(pass)) {}
    TINY_LLM_DEFAULT_COPY_MOVE(Pass);
    ~Pass() override = default;

    void run(Graph &g, WeightManagerWrapper &w) override {
      pass_concept.run(g, w);
    }
    T pass_concept;
  };

  std::vector<std::unique_ptr<PassConcept>> pipeline_;

public:
  template <typename T> void add_pass(T &&pass) {
    pipeline_.emplace_back(std::make_unique<Pass<T>>(std::forward<T>(pass)));
  }

  void run(Graph &g, WeightManagerWrapper &w) {
    for (auto &pass : pipeline_) {
      pass->run(g, w);
    }
  }
};

class PruningPass {
public:
  void run(Graph &g, WeightManagerWrapper &w);
};
} // namespace tiny_llm
