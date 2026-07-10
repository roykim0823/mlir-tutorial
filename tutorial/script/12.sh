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

{ step "5. Step: watch the solver work (12-global-optimization-and-dataflow-analysis.md)"; } 2>/dev/null
# The squaring chain, the all-adds chain, the branching star exhibit, and the
# remultiplying control experiment are all functions of this one test file.
$TUTORIAL_OPT --noisy-reduce-noise-optimizer $TESTS/noisy_reduce_noise.mlir

# skipped: bazel test //tests:noisy_reduce_noise.mlir.test //tests:noisy_syntax.mlir.test (build-system test runner, see §5)
# skipped: llvm-lit -sv build-ninja/tests --filter noisy (CMake test runner; build-ninja is stale, see §5)

set +x # turn off command echoing
