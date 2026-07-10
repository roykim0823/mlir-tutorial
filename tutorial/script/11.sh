#!/bin/bash

# run relative to this script's location (tutorial/script/), so it works from any cwd
cd "$(dirname "$0")" || exit 1

# print a section banner. The { ...; } 2>/dev/null redirections keep set -x
# trace noise out of the output: inside the function for its own commands,
# and at each call site ({ step "..."; } 2>/dev/null) for the call itself.
step() { { set +x; } 2>/dev/null; echo; echo "### $* ###"; set -x; }

set -x  # turn on command echoing
TESTS="../../tests"
TUTORIAL_OPT="${TUTORIAL_OPT:-../../bazel-bin/tools/tutorial-opt}"
# Adaptation: $TUTORIAL_OPT is built against the Bazel-pinned LLVM, which emits
# newer llvm-dialect syntax (e.g. `getelementptr inbounds|nuw`) that the
# Homebrew LLVM 20 mlir-translate on PATH cannot parse. Use the version-matched
# mlir-translate that the Bazel build produced alongside tutorial-opt.
MLIR_TRANSLATE="../../bazel-bin/external/+_repo_rules+llvm-project/mlir/mlir-translate"
# All generated artifacts (LLVM IR, object files, the executable) go to a temp
# dir -- the markdown says to run section 4 "from a scratch directory".
SCRATCH=$(mktemp -d)

{ step "2. The pipeline, stage by stage (11-lowering-through-llvm.md)"; } 2>/dev/null
# The running example @add_ct (degree 4), watched at three points along the
# pipeline. Stage boundaries follow the markdown's stage numbering.
cat > $SCRATCH/add_ct.mlir <<'EOF'
func.func @add_ct(%arg : !poly.poly<4>) -> !poly.poly<4> {
  %0 = poly.constant dense<[1, 2, 3, 4]> : tensor<4xi32> : !poly.poly<4>
  %1 = poly.add %0, %arg : !poly.poly<4>
  return %1 : !poly.poly<4>
}
EOF
# Stage 2: after poly-to-standard + elementwise/tensor -> linalg
$TUTORIAL_OPT $SCRATCH/add_ct.mlir \
  --pass-pipeline='builtin.module(poly-to-standard,canonicalize,convert-elementwise-to-linalg,convert-tensor-to-linalg)'
# Stage 3: plus one-shot bufferization (tensors become memrefs, the constant
# becomes a memref.global)
$TUTORIAL_OPT $SCRATCH/add_ct.mlir \
  --pass-pipeline='builtin.module(poly-to-standard,canonicalize,convert-elementwise-to-linalg,convert-tensor-to-linalg,one-shot-bufferize{bufferize-function-boundaries=true})'
# Stages 4-5, i.e. the whole pipeline: pure llvm dialect; note the exploded
# five-argument memref calling convention in @add_ct's signature
$TUTORIAL_OPT $SCRATCH/add_ct.mlir --poly-to-llvm

{ step "3. Step: run the pipeline"; } 2>/dev/null
$TUTORIAL_OPT $TESTS/poly_to_llvm.mlir --poly-to-llvm

{ step "4. Step: out of MLIR, into an executable"; } 2>/dev/null
# 4.1. MLIR -> LLVM IR (textual): leave MLIR-land
$TUTORIAL_OPT $TESTS/poly_to_llvm.mlir --poly-to-llvm \
  | $MLIR_TRANSLATE --mlir-to-llvmir > $SCRATCH/poly_fn.ll

# 4.2. LLVM IR -> native object file
llc --relocation-model=pic -filetype=obj < $SCRATCH/poly_fn.ll > $SCRATCH/poly_fn.o

# 4.3. Compile the C caller; link the two
clang -c $TESTS/poly_to_llvm_main.c -o $SCRATCH/main.o
clang $SCRATCH/main.o $SCRATCH/poly_fn.o -o $SCRATCH/a.out

# 4.4. Run it (expected output: Result: 351)
$SCRATCH/a.out

# 5. The lit test: five RUN lines to a running binary (11-lowering-through-llvm.md)
# skipped: bazel test //tests:poly_to_llvm.mlir.test //tests:poly_to_llvm_eval.mlir.test (build-system test runner, see §5)
# skipped: llvm-lit -sv build-ninja/tests --filter poly_to_llvm (CMake test runner; build-ninja is stale, see §5)

rm -rf $SCRATCH
set +x # turn off command echoing
