load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

package(default_visibility = ["//:__pkg__"])

cc_binary(
    name = "benchmarks",
    srcs = ["main.cpp"] + glob(["cpu/*"]) + select({
        "//platforms/device:cuda": glob(["cuda/*"]),
        "//conditions:default": [],
    }),
    deps = [
        "//tiny_llm/ops/cpu:cpu_ops",
        "//tiny_llm/tensor",
        "@google_benchmark//:benchmark",
    ] + select({
        "//platforms/device:cuda": [
            "//tiny_llm/device_managers/cuda:cuda_context",
            "//tiny_llm/ops/cuda:cuda_ops",
        ],
        "//conditions:default": [],
    }),
)
