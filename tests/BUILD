load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(default_visibility = ["//:__pkg__"])

cc_test(
    name = "tests",
    srcs = [
        "examples/test_group.cpp",
        "main_gtest.cpp",
    ] + glob([
        "device_managers/test_cpu*",
    ]) + select({
        "//platforms/device:cuda": glob(["device_managers/test_cuda*"]),
        "//conditions:default": [],
    }),
    deps = [
        "//tiny_llm/common:log_and_excepts",
        "//tiny_llm/device_managers:cpu_allocator",
        "@googletest//:gtest",
        "@spdlog",
    ] + select({
        "//platforms/device:cuda": ["//tiny_llm/device_managers:cuda_allocator"],
        "//conditions:default": [],
    }),
)
