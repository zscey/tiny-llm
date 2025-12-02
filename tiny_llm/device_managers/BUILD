load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//:__subpackages__"])

cc_library(
    name = "buffer",
    srcs = ["buffer.cpp"],
    hdrs = ["buffer.hpp"],
    deps = [
        "//tiny_llm/common:common_macros",
    ],
)

cc_library(
    name = "cpu_allocator",
    srcs = ["cpu_allocator.cpp"],
    hdrs = ["cpu_allocator.hpp"],
    deps = [
        ":buffer",
        "//tiny_llm/common:log_and_excepts",
    ],
)

cc_library(
    name = "cuda_context",
    srcs = ["cuda_context.cpp"],
    hdrs = ["cuda_context.hpp"],
    local_defines = [
        "CUDA_STREAM_POOL_SIZE=5",
        "MAX_CUDA_DEVICE_NUM=8",
    ],
    deps = [
        ":buffer",
        "//tiny_llm/common:log_and_excepts",
        "@local_cuda//:cuda_runtime",
    ],
)

cc_library(
    name = "cuda_guards",
    srcs = ["cuda_guards.cpp"],
    hdrs = ["cuda_guards.hpp"],
    deps = [
        ":cuda_context",
    ],
)

cc_library(
    name = "cuda_allocator",
    srcs = ["cuda_allocator.cpp"],
    hdrs = ["cuda_allocator.hpp"],
    deps = [
        ":cuda_guards",
    ],
)
