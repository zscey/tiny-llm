load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(default_visibility = ["//:__pkg__"])

cc_test(
    name = "examples_group",
    srcs = [
        "examples/test_group.cpp",
        "main_gtest.cpp",
    ],
    deps = [
        "@googletest//:gtest",
        "@spdlog",
    ],
)
