#!/bin/bash

# run relative to this script's location (tutorial/script/), so it works from any cwd
cd "$(dirname "$0")" || exit 1

set -x  # turn on command echoing
TESTS="../../tests"
TUTORIAL_OPT="${TUTORIAL_OPT:-../../bazel-bin/tools/tutorial-opt}"

# Prerequisites (03-writing-our-first-pass.md)
# skipped: bazel build //tools:tutorial-opt (build command; $TUTORIAL_OPT is already built)
# skipped: cmake --build build-ninja --target tutorial-opt (CMake build variant)

# 3. tutorial-opt: an `opt` tool of our own (03-writing-our-first-pass.md)
$TUTORIAL_OPT --help | grep -E 'affine-full-unroll|mul-to-add'

# 6. Step: run the unrolling pass (03-writing-our-first-pass.md)
$TUTORIAL_OPT $TESTS/affine_loop_unroll.mlir --affine-full-unroll

# 7. The same pass with the pattern rewrite engine (03-writing-our-first-pass.md)
$TUTORIAL_OPT $TESTS/affine_loop_unroll.mlir --affine-full-unroll-rewrite

# 8. A real rewrite: multiplications into additions (03-writing-our-first-pass.md)
$TUTORIAL_OPT $TESTS/mul_to_add.mlir --mul-to-add

# 11. Step: test the passes (03-writing-our-first-pass.md)
# skipped: bazel test //tests:affine_loop_unroll.mlir.test //tests:mul_to_add.mlir.test (bazel test; see §11)
# skipped: llvm-lit -sv build-ninja/tests --filter 'affine_loop_unroll|mul_to_add' (llvm-lit; build-ninja is stale; see §11)

# 4. Anatomy of a pass (03-writing-our-first-pass.md)
# The tutorial-exact (hand-written PassWrapper, no tablegen) version of the
# same three passes lives in tutorial/code/03 as its own binary; same flags,
# same outputs on this tutorial's tests.
# skipped: bazel build //tutorial/code/03:tutorial-opt-03 (build command; $TUTORIAL_OPT_03 is already built)
TUTORIAL_OPT_03="${TUTORIAL_OPT_03:-../../bazel-bin/tutorial/code/03/tutorial-opt-03}"
$TUTORIAL_OPT_03 --help | grep -E 'affine-full-unroll|mul-to-add'
$TUTORIAL_OPT_03 $TESTS/affine_loop_unroll.mlir --affine-full-unroll
$TUTORIAL_OPT_03 $TESTS/affine_loop_unroll.mlir --affine-full-unroll-rewrite
$TUTORIAL_OPT_03 $TESTS/mul_to_add.mlir --mul-to-add

set +x # turn off command echoing
