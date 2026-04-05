load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//:__subpackages__"])

cc_library(
    name = "logit_processors",
    hdrs = ["logit_processors.hpp"],
    deps = ["//tiny_llm/tensor"],
)

cc_library(
    name = "sort_processors",
    srcs = ["sort_processors.cpp"],
    hdrs = ["sort_processors.hpp"],
    deps = [
        ":logit_processors",
        "//tiny_llm/common:log_and_excepts",
    ],
)
