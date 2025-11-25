load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//:__subpackages__"])

cc_library(
    name = "common_macros",
    hdrs = ["common_macros.hpp"],
)

cc_library(
    name = "log_and_excepts",
    hdrs = ["log_and_excepts.hpp"],
    deps = ["@spdlog"],
)
