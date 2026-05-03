#pragma once

#include "tiny_llm/common/construct_macros.hpp"
#include "tiny_llm/graph/graph.hpp"
#include <concepts>
#include <memory>

namespace tiny_llm {
template <typename T>
concept GraphPass = std::move_constructible<T> && requires(T t, Graph &g) {
  { t.run(g) } -> std::same_as<void>;
};

class PassManager {
  struct Concept {
    Concept() = default;
    TINY_LLM_DEFAULT_COPY_MOVE(Concept);
    virtual ~Concept() = default;

    virtual void run(Graph &g) = 0;
  };

  template <GraphPass T> struct Container final : public Concept {
    explicit Container(T &&pass) : pass_concept(std::move(pass)) {}
    TINY_LLM_DEFAULT_COPY_MOVE(Container);
    ~Container() override = default;

    void run(Graph &g) override { pass_concept.run(g); }
    T pass_concept;
  };

  std::vector<std::unique_ptr<Concept>> pipeline_;

public:
  template <typename T>
    requires(!std::is_same_v<std::decay_t<T>, PassManager>)
  void add_pass(T &&pass) {
    pipeline_.emplace_back(
        std::make_unique<Container<std::decay_t<T>>>(std::forward<T>(pass)));
  }

  void run(Graph &g) {
    for (auto &pass : pipeline_) {
      pass->run(g);
    }
  }
};

class DCEPass {
public:
  void run(Graph &g);
};
} // namespace tiny_llm
