load("@host_platform//:constraints.bzl", "HOST_CONSTRAINS")

package(default_visibility = ["//visibility:public"])

platform(
    name = "host_platform",
    constraint_values = HOST_CONSTRAINS,
)

platform(
    name = "x86_linux_avx2_gcc13",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
        "//platforms/isa:avx2",
        "//platforms/compiler_version:gcc13",
    ],
    flags = [
        "--@rules_cuda//cuda:enable=False",
    ],
)
