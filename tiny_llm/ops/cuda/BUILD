load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cuda//cuda:defs.bzl", "cuda_library")

package(default_visibility = ["//visibility:private"])

cc_library(
    name = "cuda_op_common",
    hdrs = [
        "cuda_op_common.hpp",
    ],
    deps = [
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

cc_library(
    name = "cuda_ops",
    visibility = ["//:__subpackages__"],
    deps = [":silu"],
)
