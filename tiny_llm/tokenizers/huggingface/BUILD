load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_rust//rust:defs.bzl", "rust_static_library")

package(default_visibility = ["//:__subpackages__"])

rust_static_library(
    name = "rust_tokenizers",
    srcs = ["src/lib.rs"],
    edition = "2021",
    visibility = ["//visibility:private"],
    deps = [
        "@tokenizers_deps//:tokenizers",
    ],
)

cc_library(
    name = "tokenizers",
    srcs = ["tokenizers.cpp"],
    hdrs = ["tokenizers.hpp"],
    deps = [
        ":rust_tokenizers",
        "//tiny_llm/common:log_and_excepts",
        "//tiny_llm/tokenizers",
    ],
)
