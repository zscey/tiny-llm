load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//:__subpackages__"])

cc_library(
    name = "pipeline",
    srcs = ["pipeline.cpp"],
    hdrs = ["pipeline.hpp"],
    deps = [
        "//tiny_llm/logit_processors:argmax_processor",
        "//tiny_llm/logit_processors:sample_processors",
        "//tiny_llm/logit_processors:sort_processors",
        "//tiny_llm/logit_processors:temperature_processor",
        "//tiny_llm/parser:llama_parser",
        "//tiny_llm/runtime/cuda:cuda_runtime",
        "//tiny_llm/tensor",
        "//tiny_llm/tokenizers/huggingface:tokenizers",
        "//tiny_llm/weight_managers/safetensors:weight_manager",
    ],
)
