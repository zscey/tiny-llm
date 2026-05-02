load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//:__subpackages__"])

cc_library(
    name = "runfile",
    srcs = ["runfile.cpp"],
    hdrs = ["runfile.hpp"],
    deps = [
        "//tiny_llm/common",
        "@bazel_tools//tools/cpp/runfiles",
    ],
)

cc_library(
    name = "visitor",
    hdrs = ["visitor.hpp"],
)
