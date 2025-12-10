load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//:__subpackages__"])

cc_library(
    name = "cpu_allocator",
    srcs = ["cpu_allocator.cpp"],
    hdrs = ["cpu_allocator.hpp"],
    deps = [
        "//tiny_llm/common:log_and_excepts",
        "//tiny_llm/device_managers:buffer",
    ],
)
