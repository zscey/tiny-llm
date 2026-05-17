load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//tiny_llm/ops:__subpackages__"])

cc_library(
    name = "arithmetic_type",
    hdrs = [
        "arithmetic_type.hpp",
    ],
)

cc_library(
    name = "details",
    hdrs = ["details.hpp"],
    deps = select({
        "//platforms/device:cuda": ["@cuda//:cuda_runtime"],
        "//conditions:default": [],
    }),
)
