#!/bin/bash

# run relative to this script's location (tutorial/script/), so it works from any cwd
cd "$(dirname "$0")" || exit 1

set -x  # turn on command echoing
TESTS="../../tests"
TUTORIAL_OPT="${TUTORIAL_OPT:-../../bazel-bin/tools/tutorial-opt}"

# 3. The pattern in C++ (the article's version) (09-canonicalizers-and-drr.md)
# The article-era C++ DifferenceOfSquares pattern is kept buildable in
# tutorial/code/09 as its own pass. Same rewrite as the DRR version below;
# the other-uses test function is left alone by the hasOneUse guard.
# skipped: bazel build //tutorial/code/09:tutorial-opt-09 (build command; $TUTORIAL_OPT_09 is already built)
TUTORIAL_OPT_09="${TUTORIAL_OPT_09:-../../bazel-bin/tutorial/code/09/tutorial-opt-09}"
$TUTORIAL_OPT_09 $TESTS/poly_canonicalize.mlir --difference-of-squares

# 4. The same pattern in DRR (the repo's version) — "From DRR to C++" (09-canonicalizers-and-drr.md)
# run the -gen-rewriters tablegen backend over PolyPatterns.td and skim the output
# (paths adapted: markdown runs from the repo root; lib/... -> ../../lib/...)
mlir-tblgen --gen-rewriters -I /opt/homebrew/opt/llvm@20/include \
  -I ../../lib/Dialect/Poly ../../lib/Dialect/Poly/PolyPatterns.td

# 5. Step: watch the canonicalizations (09-canonicalizers-and-drr.md)
# the four test functions, taken in turn with --canonicalize
# (command given inline in the markdown's section 5 prose)
cat $TESTS/poly_canonicalize.mlir

$TUTORIAL_OPT --canonicalize $TESTS/poly_canonicalize.mlir

# skipped: bazel test //tests:poly_canonicalize.mlir.test (build-system command, see §5)
# skipped: llvm-lit -sv build-ninja/tests --filter poly_canonicalize (build-system command, see §5)

set +x # turn off command echoing
