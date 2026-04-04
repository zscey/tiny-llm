load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//visibility:private"])

cc_library(
    name = "cpu_op_common",
    hdrs = [
        "cpu_op_common.hpp",
    ],
    defines = select({
        "//platforms/isa:avx2": ["TINY_LLM_CPU_OPS_WITH_AVX2"],
        "//conditions:default": [],
    }),
    deps = ["//tiny_llm/thread_pool"],
)

cc_library(
    name = "arithmetic",
    srcs = ["arithmetic.cpp"],
    hdrs = ["arithmetic.hpp"],
    deps = [
        "cpu_op_common",
        "//tiny_llm/ops:arithmetic_type",
    ],
)

cc_library(
    name = "cpu_ops",
    visibility = ["//:__subpackages__"],
    deps = ["arithmetic"],
)
