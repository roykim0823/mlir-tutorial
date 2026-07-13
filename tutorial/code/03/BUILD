# Frozen code for Chapter 3 (Writing Our First Pass): the hand-written
# PassWrapper form of the passes, before the tablegen migration of Chapter 4.
# The canonical, current versions live in lib/Transform/{Affine,Arith}.

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "AffineFullUnroll",
    srcs = ["AffineFullUnroll.cpp"],
    hdrs = ["AffineFullUnroll.h"],
    deps = [
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "AffineFullUnrollPatternRewrite",
    srcs = ["AffineFullUnrollPatternRewrite.cpp"],
    hdrs = ["AffineFullUnrollPatternRewrite.h"],
    deps = [
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "MulToAdd",
    srcs = ["MulToAdd.cpp"],
    hdrs = ["MulToAdd.h"],
    deps = [
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_binary(
    name = "tutorial-opt-03",
    srcs = ["tutorial-opt.cpp"],
    deps = [
        ":AffineFullUnroll",
        ":AffineFullUnrollPatternRewrite",
        ":MulToAdd",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Pass",
    ],
)
