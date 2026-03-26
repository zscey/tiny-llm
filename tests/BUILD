load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(default_visibility = ["//:__pkg__"])

cc_test(
    name = "tests",
    srcs = [
        "examples/test_group.cpp",
        "main_gtest.cpp",
    ] + glob([
        "device_managers/cpu/*",
        "graph/*",
        "tensor/*",
        "weight_managers/*",
        "runtime/*",
        "tokenizers/*",
        "utils/*",
    ]) + select({
        "//platforms/device:cuda": glob([
            "device_managers/cuda/*",
            "runtime/cuda/*",
            "ops/cuda/*",
        ]),
        "//conditions:default": [],
    }),
    data = ["//tests/datas"],
    local_defines = select({
        "//platforms/device:cuda": ["TESTS_WITH_CUDA"],
        "//conditions:default": [],
    }),
    deps = [
        "//tiny_llm/common:log_and_excepts",
        "//tiny_llm/device_managers/cpu:cpu_allocator",
        "//tiny_llm/graph:graph_optimizer",
        "//tiny_llm/runtime:greedy_memory_planer",
        "//tiny_llm/tensor",
        "//tiny_llm/tokenizers/huggingface:tokenizers",
        "//tiny_llm/utils:runfile",
        "//tiny_llm/weight_managers/safetensors:weight_manager",
        "@googletest//:gtest",
        "@spdlog",
    ] + select({
        "//platforms/device:cuda": [
            "//tiny_llm/device_managers/cuda:cuda_allocator",
            "//tiny_llm/device_managers/cuda:cuda_device_infos",
            "//tiny_llm/runtime/cuda:cuda_runtime",
        ],
        "//conditions:default": [],
    }),
)
