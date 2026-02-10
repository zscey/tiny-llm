load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//:__subpackages__"])

cc_library(
    name = "param",
    srcs = ["param.cpp"],
    hdrs = ["param.hpp"],
    deps = ["//tiny_llm/utils:visitor"],
)

cc_library(
    name = "graph",
    srcs = ["graph.cpp"],
    hdrs = ["graph.hpp"],
    deps = [
        ":param",
        "//tiny_llm/common:log_and_excepts",
        "//tiny_llm/tensor",
    ],
)

cc_library(
    name = "graph_optimizer",
    srcs = ["graph_optimizer.cpp"],
    hdrs = ["graph_optimizer.hpp"],
    deps = [
        ":graph",
        "//tiny_llm/weight_managers",
    ],
)
