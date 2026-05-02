load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//:__subpackages__"])

cc_library(
    name = "common",
    hdrs = ["common.hpp"],
    deps = ["//tiny_llm/tensor"],
)

cc_library(
    name = "memory_planer",
    hdrs = [
        "memory_planer.hpp",
    ],
    deps = [
        "//tiny_llm/common",
    ],
)

cc_library(
    name = "greedy_memory_planer",
    srcs = ["greedy_memory_planer.cpp"],
    hdrs = ["greedy_memory_planer.hpp"],
    deps = [
        "memory_planer",
    ],
)

cc_library(
    name = "iruntime",
    hdrs = ["iruntime.hpp"],
    deps = [
        "//tiny_llm/tensor",
    ],
)
