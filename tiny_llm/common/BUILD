load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//:__subpackages__"])

cc_library(
    name = "common",
    hdrs = [
        "checks.hpp",
        "construct_macros.hpp",
        "exception.hpp",
    ] + select({
        "//platforms/device:cuda": ["cuda_checks.hpp"],
        "//conditions:default": [],
    }),
    deps = [
        "@spdlog",
    ] + select({
        "//platforms/device:cuda": ["@cuda//:cuda_runtime"],
        "//conditions:default": [],
    }),
)
