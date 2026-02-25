load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//:__subpackages__"])

cc_library(
    name = "cuda_kernels",
    srcs = ["cuda_kernels.cpp"],
    hdrs = ["cuda_kernels.hpp"],
    deps = [
        "//tiny_llm/common:log_and_excepts",
        "//tiny_llm/ops/cuda:cuda_ops",
        "//tiny_llm/runtime:common",
    ],
)

cc_library(
    name = "cuda_plan",
    srcs = ["cuda_plan.cpp"],
    hdrs = ["cuda_plan.hpp"],
    deps = [
        ":cuda_kernels",
        "//tiny_llm/graph",
        "//tiny_llm/utils:visitor",
    ],
)

cc_library(
    name = "cuda_runtime",
    srcs = ["cuda_runtime.cpp"],
    hdrs = ["cuda_runtime.hpp"],
    deps = [
        ":cuda_plan",
        "//tiny_llm/common:log_and_excepts",
        "//tiny_llm/device_managers/cuda:cuda_allocator",
        "//tiny_llm/device_managers/cuda:cuda_context",
        "//tiny_llm/runtime:greedy_memory_planer",
        "//tiny_llm/runtime:iruntime",
        "//tiny_llm/weight_managers",
    ],
)
