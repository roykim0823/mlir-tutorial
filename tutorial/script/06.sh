#!/bin/bash

# run relative to this script's location (tutorial/script/), so it works from any cwd
cd "$(dirname "$0")" || exit 1

set -x  # turn on command echoing
TESTS="../../tests"
TUTORIAL_OPT="${TUTORIAL_OPT:-../../bazel-bin/tools/tutorial-opt}"

# 3. Step: watch the upstream passes work (06-using-traits.md)

# Common subexpression elimination
$TUTORIAL_OPT -cse $TESTS/cse.mlir

# Loop-invariant code motion
$TUTORIAL_OPT --loop-invariant-code-motion $TESTS/code_motion.mlir

# Control-flow sinking
$TUTORIAL_OPT -control-flow-sink $TESTS/control_flow_sink.mlir

# skipped: bazel test //tests:cse.mlir.test //tests:code_motion.mlir.test //tests:control_flow_sink.mlir.test (bazel test; see §3)
# skipped: llvm-lit -sv build-ninja/tests --filter 'cse|code_motion|control_flow_sink' (llvm-lit; build-ninja is stale; see §3)

# 3. The dog that didn't bark (06-using-traits.md)
# poly.eval has no Pure trait, so CSE must not deduplicate it: both evals survive.
SCRATCH=$(mktemp -d)
cat > $SCRATCH/dup_eval.mlir <<'EOF'
func.func @dup_eval(%p: !poly.poly<10>, %x: i32) -> i32 {
  %0 = poly.eval %p, %x : (!poly.poly<10>, i32) -> i32
  %1 = poly.eval %p, %x : (!poly.poly<10>, i32) -> i32
  %2 = arith.addi %0, %1 : i32
  return %2 : i32
}
EOF
$TUTORIAL_OPT -cse $SCRATCH/dup_eval.mlir
rm -rf $SCRATCH

set +x # turn off command echoing
