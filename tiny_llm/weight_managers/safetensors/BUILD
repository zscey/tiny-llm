load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_rust//rust:defs.bzl", "rust_static_library")

package(default_visibility = ["//:__subpackages__"])

rust_static_library(
    name = "rust_safetensors",
    srcs = ["src/lib.rs"],
    edition = "2021",
    visibility = ["//visibility:private"],
    deps = [
        "@safetensors_deps//:memmap2",
        "@safetensors_deps//:safetensors",
    ],
)

cc_library(
    name = "weight_manager",
    srcs = ["weight_manager.cpp"],
    hdrs = ["weight_manager.hpp"],
    deps = [
        ":rust_safetensors",
        "//tiny_llm/common",
        "//tiny_llm/weight_managers",
    ],
)
