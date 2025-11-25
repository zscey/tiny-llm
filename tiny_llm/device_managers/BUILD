load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//:__subpackages__"])

cc_library(
    name = "buffer",
    srcs = ["buffer.cpp"],
    hdrs = ["buffer.hpp"],
)

cc_library(
    name = "cpu_allocator",
    srcs = ["cpu_allocator.cpp"],
    hdrs = ["cpu_allocator.hpp"],
    deps = [
        ":buffer",
        "//tiny_llm/common:common_macros",
        "//tiny_llm/common:log_and_excepts",
    ],
)
