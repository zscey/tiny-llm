load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//:__subpackages__"])

cc_library(
    name = "cuda_device_guard",
    srcs = ["cuda_device_guard.cpp"],
    hdrs = ["cuda_device_guard.hpp"],
    deps = [
        "//tiny_llm/common:common_macros",
        "//tiny_llm/common:log_and_excepts",
        "@local_cuda//:cuda_runtime",
    ],
)

cc_library(
    name = "cuda_context",
    srcs = ["cuda_context.cpp"],
    hdrs = ["cuda_context.hpp"],
    local_defines = [
        "CUDA_STREAM_POOL_SIZE=32",
        "MAX_CUDA_DEVICE_NUM=8",
    ],
    deps = [
        ":cuda_device_guard",
        "//tiny_llm/device_managers:buffer",
    ],
)

cc_library(
    name = "cuda_allocator",
    srcs = ["cuda_allocator.cpp"],
    hdrs = ["cuda_allocator.hpp"],
    deps = [
        ":cuda_context",
    ],
)

cc_library(
    name = "cuda_device_infos",
    srcs = ["cuda_device_infos.cpp"],
    hdrs = ["cuda_device_infos.hpp"],
    deps = [
        "//tiny_llm/common:log_and_excepts",
        "@local_cuda//:cuda_runtime",
    ],
)
