#!/bin/bash

# run relative to this script's location (tutorial/script/), so it works from any cwd
cd "$(dirname "$0")" || exit 1

set -x  # turn on command echoing
TESTS="../../tests"
TUTORIAL_OPT="${TUTORIAL_OPT:-../../bazel-bin/tools/tutorial-opt}"
# Include path for upstream .td files, as used in the markdown
LLVM_INCLUDE="/opt/homebrew/opt/llvm@20/include"

SCRATCH=$(mktemp -d)

# 1. What `poly` represents, concretely (05-defining-a-new-dialect.md)
# The "tiny poly program" from section 1: build p = 1 + 2x + 3x^2, square it,
# evaluate at x = 7. Wrapped in a function so it parses standalone.
cat > $SCRATCH/poly_square_eval.mlir <<'EOF'
func.func @poly_square_eval() -> i32 {
  %coeffs = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
  %p  = poly.from_tensor %coeffs : tensor<3xi32> -> !poly.poly<10>    // p = 1 + 2x + 3x^2
  %sq = poly.mul %p, %p : !poly.poly<10>                              // p^2 = 1 + 4x + 10x^2 + 12x^3 + 9x^4
  %c7 = arith.constant 7 : i32
  %v  = poly.eval %sq, %c7 : (!poly.poly<10>, i32) -> i32             // p^2(7) = 26244
  return %v : i32
}
EOF
# Round-trip it (nothing is computed at this point in the series; the comments
# above are math annotations, not compiler output).
$TUTORIAL_OPT $SCRATCH/poly_square_eval.mlir
# Foreshadowing Tutorial 7: --canonicalize folds the mul, producing exactly the
# p^2 coefficients from the comment (dense<[1, 4, 10, 12, 9]>). The eval is NOT
# folded -- EvalOp has no folder -- so 26244 stays a math annotation.
$TUTORIAL_OPT $SCRATCH/poly_square_eval.mlir --canonicalize

# 2. The dialect shell (05-defining-a-new-dialect.md)
# (markdown paths lib/Dialect/Poly/... rewritten relative to tutorial/)
mlir-tblgen --gen-dialect-decls -I "$LLVM_INCLUDE" \
  -I ../../lib/Dialect/Poly ../../lib/Dialect/Poly/PolyDialect.td

# 7. Step: exercise the syntax (05-defining-a-new-dialect.md)
$TUTORIAL_OPT $TESTS/poly_syntax.mlir

# skipped: bazel test //tests:poly_syntax.mlir.test (bazel test; see §7)
# skipped: llvm-lit -sv build-ninja/tests --filter poly_syntax (llvm-lit; build-ninja is stale; see §7)

rm -rf $SCRATCH
set +x # turn off command echoing
