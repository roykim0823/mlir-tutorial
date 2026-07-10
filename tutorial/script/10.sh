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

{ step "3. Conversion patterns — \"Multiplication: creating a loop nest\" (10-dialect-conversion.md)"; } 2>/dev/null
# the degree-4 example from the markdown, as a scratch file
cat > $SCRATCH/lower_mul.mlir <<'EOF'
func.func @lower_mul(%p: !poly.poly<4>, %q: !poly.poly<4>) -> !poly.poly<4> {
  %0 = poly.mul %p, %q : !poly.poly<4>
  return %0 : !poly.poly<4>
}
EOF

$TUTORIAL_OPT --poly-to-standard $SCRATCH/lower_mul.mlir

{ step "5. Step: run the conversion"; } 2>/dev/null
cat $TESTS/poly_to_standard.mlir

$TUTORIAL_OPT --poly-to-standard $TESTS/poly_to_standard.mlir

# skipped: bazel test //tests:poly_to_standard.mlir.test (build-system command, see §5)
# skipped: llvm-lit -sv build-ninja/tests --filter poly_to_standard (build-system command, see §5)

{ step "6. Step: read a legality failure"; } 2>/dev/null
# ConvertEval doesn't handle complex points, so feed it one
# (command implied by the markdown; the error output shown there is real)
cat > $SCRATCH/complex_eval.mlir <<'EOF'
func.func @f(%p: !poly.poly<10>, %z: complex<f64>) -> complex<f64> {
  %0 = poly.eval %p, %z : (!poly.poly<10>, complex<f64>) -> complex<f64>
  return %0 : complex<f64>
}
EOF

# expected to FAIL: failed to legalize unresolved materialization from ('i32')
# to ('complex<f64>') that remained live after conversion
$TUTORIAL_OPT --poly-to-standard $SCRATCH/complex_eval.mlir

rm -rf $SCRATCH
set +x # turn off command echoing
