#!/bin/bash

# run relative to this script's location (tutorial/script/), so it works from any cwd
cd "$(dirname "$0")" || exit 1

set -x  # turn on command echoing
TESTS="../../tests"
TUTORIAL_OPT="${TUTORIAL_OPT:-../../bazel-bin/tools/tutorial-opt}"
# Include path for upstream .td files (mlir/Pass/PassBase.td), as used in the markdown
LLVM_INCLUDE="/opt/homebrew/opt/llvm@20/include"
# Scratch dir for temp .td files (markdown writes /tmp/extra.td; keep scratch out of the repo)
SCRATCH=$(mktemp -d)

# 3. Step: run the generator by hand (04-using-tablegen-for-passes.md)
# (markdown path lib/Transform/Affine/Passes.td rewritten relative to tutorial/)
mlir-tblgen --gen-pass-decls -name Affine \
  -I "$LLVM_INCLUDE" \
  ../../lib/Transform/Affine/Passes.td | wc -l

# 6. Step: watch a change flow through (04-using-tablegen-for-passes.md)
cat > $SCRATCH/extra.td <<'EOF'
include "mlir/Pass/PassBase.td"
def AffineCountLoops : Pass<"affine-count-loops"> {
  let summary = "Count affine loops";
}
EOF
mlir-tblgen --gen-pass-decls -name Affine \
  -I "$LLVM_INCLUDE" $SCRATCH/extra.td | grep AFFINECOUNTLOOPS

rm -rf $SCRATCH
set +x # turn off command echoing
