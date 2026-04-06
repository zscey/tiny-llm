load("@bazel_output_base_util//:defs.bzl", "OUTPUT_BASE")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_compdb//:defs.bzl", "compilation_database")

compilation_database(
    name = "compdb",
    testonly = True,
    output_base = OUTPUT_BASE,
    targets = [
        "//tests",
        "//benchmarks",
    ],
)

cc_binary(
    name = "main",
    srcs = ["main.cpp"],
    deps = [
        "//tiny_llm/device_managers/cuda:cuda_device_guard",
        "//tiny_llm/pipeline",
        "@gflags",
    ],
)
