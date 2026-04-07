load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

cc_binary(
    name = "main",
    srcs = ["main.cpp"],
    deps = [
        "//tiny_llm/device_managers/cuda:cuda_device_guard",
        "//tiny_llm/pipeline",
        "@gflags",
    ],
)
