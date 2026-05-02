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
        "@cuda//:cuda_runtime",
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
        "//tiny_llm/common",
        "@cuda//:cub",
        "@cuda//:libcudacxx",
    ],
)

cuda_library(
    name = "rope",
    srcs = ["rope.cu"],
    hdrs = ["rope.hpp"],
    deps = [
        ":cuda_op_common",
    ],
)

cuda_library(
    name = "embedding",
    srcs = ["embedding.cu"],
    hdrs = ["embedding.hpp"],
    deps = [
        ":cuda_op_common",
    ],
)

cuda_library(
    name = "flash_attention",
    srcs = ["flash_attention.cu"],
    hdrs = ["flash_attention.hpp"],
    deps = [
        ":cuda_op_common",
        "//tiny_llm/device_managers/cuda:cuda_device_infos",
        "@cuda//:cub",
        "@cuda//:libcudacxx",
    ],
)

cuda_library(
    name = "norm",
    srcs = ["norm.cu"],
    hdrs = ["norm.hpp"],
    deps = [
        ":cuda_op_common",
        "@cuda//:cub",
        "@cuda//:libcudacxx",
    ],
)

cuda_library(
    name = "arithmetic",
    srcs = ["arithmetic.cu"],
    hdrs = ["arithmetic.hpp"],
    deps = [
        ":cuda_op_common",
        "//tiny_llm/ops:arithmetic_type",
    ],
)

cc_library(
    name = "cuda_ops",
    visibility = ["//:__subpackages__"],
    deps = [
        ":arithmetic",
        ":embedding",
        ":flash_attention",
        ":gemm",
        ":norm",
        ":rope",
        ":silu",
    ],
)
