load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//:__subpackages__"])

cc_library(
    name = "llama_parser",
    srcs = ["llama_parser.cpp"],
    hdrs = ["llama_parser.hpp"],
    deps = [
        "//tiny_llm/common",
        "//tiny_llm/graph",
        "@nlohmann_json//:json",
    ],
)
