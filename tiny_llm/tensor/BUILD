load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//:__subpackages__"])

cc_library(
    name = "tensor",
    srcs = ["tensor.cpp"],
    hdrs = ["tensor.hpp"],
    local_defines = select({
        "//platforms/device:cuda": ["TINY_LLM_TENSOR_WITH_CUDA"],
        "//conditions:default": [],
    }),
    deps = [
        "//tiny_llm/common:log_and_excepts",
        "//tiny_llm/device_managers:buffer",
        "//tiny_llm/device_managers/cpu:cpu_allocator",
    ] + select({
        "//platforms/device:cuda": ["//tiny_llm/device_managers/cuda:cuda_allocator"],
        "//conditions:default": [],
    }),
)
