#pragma once

#include "tiny_llm/common/common_macros.hpp"
#include "tiny_llm/graph/graph.hpp"
#include <concepts>
#include <memory>

namespace tiny_llm {
template <typename T>
concept GraphPass = requires(T t, Graph &g) { t.run(g); };

class PassManager {
  struct PassConcept {
    PassConcept() = default;
    TINY_LLM_DEFAULT_COPY_MOVE(PassConcept);
    virtual ~PassConcept() = default;

    virtual void run(Graph &g) = 0;
  };

  template <typename T> struct Pass final : public PassConcept {
    explicit Pass(T pass) : pass_concept(std::move(pass)) {}
    TINY_LLM_DEFAULT_COPY_MOVE(Pass);
    ~Pass() override = default;

    void run(Graph &g) override { pass_concept.run(g); }
    T pass_concept;
  };

  std::vector<std::unique_ptr<PassConcept>> pipeline_;

public:
  template <typename T> void add_pass(T &&pass) {
    pipeline_.emplace_back(std::make_unique<Pass<T>>(std::forward<T>(pass)));
  }

  void run(Graph &g) {
    for (auto &pass : pipeline_) {
      pass->run(g);
    }
  }
};

class PruningPass {
public:
  void run(Graph &g);
};
} // namespace tiny_llm
