#!/bin/bash

# run relative to this script's location (tutorial/script/), so it works from any cwd
cd "$(dirname "$0")" || exit 1

set -x  # turn on command echoing
TESTS="../../tests"
TUTORIAL_OPT="${TUTORIAL_OPT:-../../bazel-bin/tools/tutorial-opt}"
SCRATCH=$(mktemp -d)

# 4. Step: catch all three layers in the act (08-verifiers.md)
# The markdown shows the poly.eval lines and their real error output; the
# implied command is $TUTORIAL_OPT with no passes (verification runs after
# parsing). Each snippet is wrapped in a full function here.

# Layer 1 — ODS constraint: a floating-point point violates IntOrComplex
cat > $SCRATCH/eval_f32.mlir <<'EOF'
func.func @test(%p: !poly.poly<10>, %x: f32) -> f32 {
  %0 = poly.eval %p, %x : (!poly.poly<10>, f32) -> f32
  return %0 : f32
}
EOF

# expected to FAIL: 'poly.eval' op operand #1 must be integer or complex-type
$TUTORIAL_OPT $SCRATCH/eval_f32.mlir

# Layer 2 — trait verifier: i16 is an integer (ODS ok) but not 32 bits wide
cat > $SCRATCH/eval_i16.mlir <<'EOF'
func.func @test(%p: !poly.poly<10>, %x: i16) -> i16 {
  %0 = poly.eval %p, %x : (!poly.poly<10>, i16) -> i16
  return %0 : i16
}
EOF

# expected to FAIL: 'poly.eval' op requires each numeric operand to be a 32-bit integer
$TUTORIAL_OPT $SCRATCH/eval_i16.mlir

# Layer 3 — custom op verifier: si32 is 32 bits wide (trait ok) but not signless
cat > $SCRATCH/eval_si32.mlir <<'EOF'
func.func @test(%p: !poly.poly<10>, %x: si32) -> si32 {
  %0 = poly.eval %p, %x : (!poly.poly<10>, si32) -> si32
  return %0 : si32
}
EOF

# expected to FAIL: 'poly.eval' op argument point must be a 32-bit integer, or a complex number
$TUTORIAL_OPT $SCRATCH/eval_si32.mlir

# 5. Step: testing error messages (08-verifiers.md)
cat $TESTS/poly_verifier.mlir

# manual replay of the test's lit RUN line: tutorial-opt %s 2>%t; FileCheck %s < %t
# expected to FAIL (nonzero exit) on the tutorial-opt half: the file contains
# an invalid op on purpose; FileCheck then matches the captured diagnostic
$TUTORIAL_OPT $TESTS/poly_verifier.mlir 2>$SCRATCH/stderr.txt
FileCheck $TESTS/poly_verifier.mlir < $SCRATCH/stderr.txt
echo $?

# skipped: bazel test //tests:poly_verifier.mlir.test (build-system command, see §5)
# skipped: llvm-lit -sv build-ninja/tests --filter poly_verifier (build-system command, see §5)

rm -rf $SCRATCH
set +x # turn off command echoing
