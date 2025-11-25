load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(default_visibility = ["//:__pkg__"])

cc_test(
    name = "tests",
    srcs = [
        "examples/test_group.cpp",
        "main_gtest.cpp",
    ] + glob([
        "device_managers/*",
    ]),
    deps = [
        "//tiny_llm/common:log_and_excepts",
        "//tiny_llm/device_managers:cpu_allocator",
        "@googletest//:gtest",
        "@spdlog",
    ],
)
