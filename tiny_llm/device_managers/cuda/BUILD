load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//:__subpackages__"])

cc_library(
    name = "cuda_context",
    srcs = ["cuda_context.cpp"],
    hdrs = ["cuda_context.hpp"],
    local_defines = [
        "CUDA_STREAM_POOL_SIZE=32",
        "MAX_CUDA_DEVICE_NUM=8",
    ],
    deps = [
        "//tiny_llm/common:log_and_excepts",
        "//tiny_llm/device_managers:buffer",
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
