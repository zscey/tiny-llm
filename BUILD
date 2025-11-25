load("@bazel_output_base_util//:defs.bzl", "OUTPUT_BASE")
load("@rules_compdb//:defs.bzl", "compilation_database")

compilation_database(
    name = "compdb",
    testonly = True,
    output_base = OUTPUT_BASE,
    targets = [
        "//tests:tests",
    ],
)
