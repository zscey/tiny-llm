load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cuda//cuda:defs.bzl", "cuda_library")

package(default_visibility = ["//visibility:private"])

cc_library(
    name = "cuda_op_common",
    hdrs = [
        "cuda_op_common.hpp",
    ],
    deps = [
        "//tiny_llm/device_managers/cuda:cuda_context",
        "@local_cuda//:cuda_runtime",
    ],
)

cuda_library(
    name = "silu",
    srcs = ["silu.cu"],
    hdrs = ["silu.hpp"],
    deps = [
        ":cuda_op_common",
    ],
)

cuda_library(
    name = "gemm",
    srcs = ["gemm.cu"],
    hdrs = ["gemm.hpp"],
    deps = [
        ":cuda_op_common",
        "//tiny_llm/common:log_and_excepts",
    ],
)

cc_library(
    name = "cuda_ops",
    visibility = ["//:__subpackages__"],
    deps = [
        ":gemm",
        ":silu",
    ],
)
