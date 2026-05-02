load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//:__subpackages__"])

cc_library(
    name = "buffer",
    srcs = ["buffer.cpp"],
    hdrs = ["buffer.hpp"],
    deps = [
        "//tiny_llm/common",
    ],
)

cc_library(
    name = "copy_data",
    srcs = ["copy_data.cpp"],
    hdrs = ["copy_data.hpp"],
    local_defines = select({
        "//platforms/device:cuda": ["TINY_LLM_COPY_DATA_WITH_CUDA"],
        "//conditions:default": [],
    }),
    deps = [
        ":buffer",
    ] + select({
        "//platforms/device:cuda": ["//tiny_llm/device_managers/cuda:cuda_context"],
        "//conditions:default": [],
    }),
)
