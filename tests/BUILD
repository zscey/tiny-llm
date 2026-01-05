load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(default_visibility = ["//:__pkg__"])

cc_test(
    name = "tests",
    srcs = [
        "examples/test_group.cpp",
        "main_gtest.cpp",
    ] + glob([
        "device_managers/cpu/*",
        "graph/*",
        "tensor/*",
    ]) + select({
        "//platforms/device:cuda": glob(["device_managers/cuda/*"]),
        "//conditions:default": [],
    }),
    local_defines = select({
        "//platforms/device:cuda": ["TESTS_WITH_CUDA"],
        "//conditions:default": [],
    }),
    deps = [
        "//tiny_llm/common:log_and_excepts",
        "//tiny_llm/device_managers/cpu:cpu_allocator",
        "//tiny_llm/graph",
        "//tiny_llm/tensor",
        "@googletest//:gtest",
        "@spdlog",
    ] + select({
        "//platforms/device:cuda": ["//tiny_llm/device_managers/cuda:cuda_allocator"],
        "//conditions:default": [],
    }),
)
