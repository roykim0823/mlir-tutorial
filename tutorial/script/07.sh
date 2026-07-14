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
SCRATCH=$(mktemp -d)

{ step "1. Concepts: folding and its consumers (07-folders-and-constant-propagation.md)"; } 2>/dev/null
# the warm-up functions live in tests/sccp.mlir
cat $TESTS/sccp.mlir

$TUTORIAL_OPT -pass-pipeline="builtin.module(func.func(sccp))" $TESTS/sccp.mlir

# section 1 contrasts sccp with --canonicalize on the same function
# (command implied by the markdown; output shown there is from this run)
$TUTORIAL_OPT --canonicalize $TESTS/sccp.mlir

{ step "5. Step: watch it work"; } 2>/dev/null
# same sccp command, now looking at the second function @test_poly_sccp
$TUTORIAL_OPT -pass-pipeline="builtin.module(func.func(sccp))" $TESTS/sccp.mlir

{ step "5. Where does [1, 4, 10, 12, 9] come from?"; } 2>/dev/null
# "You can check the fold in isolation" — square.mlir scratch file
cat > $SCRATCH/square.mlir <<'EOF'
func.func @square() -> !poly.poly<10> {
  %coeffs = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
  %p = poly.from_tensor %coeffs : tensor<3xi32> -> !poly.poly<10>
  %sq = poly.mul %p, %p : !poly.poly<10>
  return %sq : !poly.poly<10>
}
EOF

$TUTORIAL_OPT --canonicalize $SCRATCH/square.mlir

# "The wraparound, verified." — wraparound.mlir scratch file
cat > $SCRATCH/wraparound.mlir <<'EOF'
func.func @wraparound() -> !poly.poly<10> {
  %c = arith.constant dense<[0, 0, 0, 0, 0, 0, 1]> : tensor<7xi32>
  %x6 = poly.from_tensor %c : tensor<7xi32> -> !poly.poly<10>
  %sq = poly.mul %x6, %x6 : !poly.poly<10>
  return %sq : !poly.poly<10>
}
EOF

$TUTORIAL_OPT --canonicalize $SCRATCH/wraparound.mlir

# "A declined fold, verified." — the null guards fire and --canonicalize
# leaves the ops exactly as written (command implied by the markdown)
cat > $SCRATCH/cant_fold.mlir <<'EOF'
func.func @cant_fold(%arg0: tensor<3xi32>) -> !poly.poly<10> {
  %0 = poly.from_tensor %arg0 : tensor<3xi32> -> !poly.poly<10>
  %1 = poly.mul %0, %0 : !poly.poly<10>
  return %1 : !poly.poly<10>
}
EOF

$TUTORIAL_OPT --canonicalize $SCRATCH/cant_fold.mlir

# skipped: bazel test //tests:sccp.mlir.test (build-system command, see §5)
# skipped: llvm-lit -sv build-ninja/tests --filter sccp (build-system command, see §5)

rm -rf $SCRATCH
set +x # turn off command echoing
