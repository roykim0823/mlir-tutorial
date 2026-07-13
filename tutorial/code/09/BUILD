# Frozen code for Chapter 9 (Canonicalizers and DRR): the original C++
# version of the DifferenceOfSquares pattern, exposed as a standalone pass.
# The canonical, current version is the DRR pattern in
# lib/Dialect/Poly/PolyPatterns.td.

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "DifferenceOfSquares",
    srcs = ["DifferenceOfSquares.cpp"],
    hdrs = ["DifferenceOfSquares.h"],
    deps = [
        "//lib/Dialect/Poly",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_binary(
    name = "tutorial-opt-09",
    srcs = ["tutorial-opt.cpp"],
    deps = [
        ":DifferenceOfSquares",
        "//lib/Dialect/Poly",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Pass",
    ],
)
