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
# Include path for upstream .td files (mlir/Pass/PassBase.td), as used in the markdown
LLVM_INCLUDE="/opt/homebrew/opt/llvm@20/include"
# Scratch dir for temp .td files (markdown writes /tmp/extra.td; keep scratch out of the repo)
SCRATCH=$(mktemp -d)

{ step "3. Step: run the generator by hand (04-using-tablegen-for-passes.md)"; } 2>/dev/null
# (markdown path lib/Transform/Affine/Passes.td rewritten relative to tutorial/)
mlir-tblgen --gen-pass-decls -name Affine \
  -I "$LLVM_INCLUDE" \
  ../../lib/Transform/Affine/Passes.td | wc -l

{ step "4. The second generated file is isomorphic (-name Arith)"; } 2>/dev/null
# Section 4's closing note: Arith's Passes.h.inc has the same shape with the
# names swapped -- regenerate it and skim the swapped names.
mlir-tblgen --gen-pass-decls -name Arith \
  -I "$LLVM_INCLUDE" \
  ../../lib/Transform/Arith/Passes.td \
  | grep -E 'GEN_PASS_DECL_|class MulToAdd|registerArithPasses' | head -8

{ step "6. Step: watch a change flow through"; } 2>/dev/null
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
