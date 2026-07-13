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

{ step "Prerequisites (03-writing-our-first-pass.md)"; } 2>/dev/null
# skipped: bazel build //tools:tutorial-opt (build command; $TUTORIAL_OPT is already built)
# skipped: cmake --build build-ninja --target tutorial-opt (CMake build variant)

{ step "3. tutorial-opt: an opt tool of our own"; } 2>/dev/null
$TUTORIAL_OPT --help | grep -E 'affine-full-unroll|mul-to-add'

{ step "6. Step: run the unrolling pass"; } 2>/dev/null
$TUTORIAL_OPT $TESTS/affine_loop_unroll.mlir --affine-full-unroll

{ step "7. The same pass with the pattern rewrite engine"; } 2>/dev/null
$TUTORIAL_OPT $TESTS/affine_loop_unroll.mlir --affine-full-unroll-rewrite

{ step "8. A real rewrite: multiplications into additions"; } 2>/dev/null
$TUTORIAL_OPT $TESTS/mul_to_add.mlir --mul-to-add

{ step "4. Anatomy of a pass: the preserved tutorial/code/03 binary is equivalent"; } 2>/dev/null
# The chapter-exact (hand-written PassWrapper, no tablegen) version of the
# same three passes lives in tutorial/code/03 as its own binary. The heads-up
# claims it registers the same flags and produces identical output on this
# chapter's tests -- verify that instead of re-running the same commands:
# the --help grep shows the same three flags (minus --mul-to-add-pdll, which
# belongs to Chapter 13's code), and each pass's output is diffed against
# $TUTORIAL_OPT's from the sections above.
# skipped: bazel build //tutorial/code/03:tutorial-opt-03 (build command; $TUTORIAL_OPT_03 is already built)
TUTORIAL_OPT_03="${TUTORIAL_OPT_03:-../../bazel-bin/tutorial/code/03/tutorial-opt-03}"
$TUTORIAL_OPT_03 --help | grep -E 'affine-full-unroll|mul-to-add'
diff <($TUTORIAL_OPT_03 $TESTS/affine_loop_unroll.mlir --affine-full-unroll) \
     <($TUTORIAL_OPT    $TESTS/affine_loop_unroll.mlir --affine-full-unroll) \
  && echo "identical output: --affine-full-unroll"
diff <($TUTORIAL_OPT_03 $TESTS/affine_loop_unroll.mlir --affine-full-unroll-rewrite) \
     <($TUTORIAL_OPT    $TESTS/affine_loop_unroll.mlir --affine-full-unroll-rewrite) \
  && echo "identical output: --affine-full-unroll-rewrite"
diff <($TUTORIAL_OPT_03 $TESTS/mul_to_add.mlir --mul-to-add) \
     <($TUTORIAL_OPT    $TESTS/mul_to_add.mlir --mul-to-add) \
  && echo "identical output: --mul-to-add"

# { step "11. Step: test the passes"; } 2>/dev/null
# skipped: bazel test //tests:affine_loop_unroll.mlir.test //tests:mul_to_add.mlir.test (bazel test; see §11)
# skipped: llvm-lit -sv build-ninja/tests --filter 'affine_loop_unroll|mul_to_add' (llvm-lit; build-ninja is stale; see §11)

set +x # turn off command echoing
