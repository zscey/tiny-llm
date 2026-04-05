load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//:__subpackages__"])

cc_library(
    name = "logit_processors",
    hdrs = ["logit_processors.hpp"],
    deps = ["//tiny_llm/tensor"],
)

cc_library(
    name = "topk_processor",
    srcs = ["topk_processor.cpp"],
    hdrs = ["topk_processor.hpp"],
    deps = [
        ":logit_processors",
        "//tiny_llm/common:log_and_excepts",
    ],
)
