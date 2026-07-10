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
{ step "2. Your lab bench: the mlir-pdll tool (13-defining-patterns-with-pdll.md)"; } 2>/dev/null
# The markdown's interface template is: mlir-pdll -x=mlir -I <path-to-mlir-includes> yourfile.pdll
# Adaptation: the LLVM submodule externals/llvm-project is not checked out;
# use the Homebrew LLVM 20 includes, which match the mlir-pdll on PATH.
LLVM_INCLUDE="$(brew --prefix llvm@20)/include"
# All scratch .pdll files go to a temp dir (the markdown says "put each
# snippet below in a scratch .pdll file").
SCRATCH=$(mktemp -d)

{ step "3.1 The smallest possible pattern"; } 2>/dev/null
cat > $SCRATCH/erase_constant.pdll <<'EOF'
#include "mlir/Dialect/Arith/IR/ArithOps.td"

Pattern => erase op<arith.constant>;
EOF
mlir-pdll -x=mlir -I $LLVM_INCLUDE $SCRATCH/erase_constant.pdll

{ step "3.2 Includes: how PDLL knows your ops"; } 2>/dev/null
# "You can watch the import happen: run -x=ast on the file and the dump
# *begins* with dozens of UserConstraintDecls synthesized from arith's ODS."
mlir-pdll -x=ast -I $LLVM_INCLUDE $SCRATCH/erase_constant.pdll

{ step "3.6 Step: grow a real pattern"; } 2>/dev/null
# Naive version -- compiles, but is semantically wrong: it rewrites x + 7 to x.
cat > $SCRATCH/add_zero_wrong.pdll <<'EOF'
#include "mlir/Dialect/Arith/IR/ArithOps.td"

Pattern EliminateAddZeroWRONG {
  let root = op<arith.addi>(x: Value, op<arith.constant>);
  replace root with x;
}
EOF
mlir-pdll -x=mlir -I $LLVM_INCLUDE $SCRATCH/add_zero_wrong.pdll

# Correct version: bind the constant's payload and check it with a native constraint.
cat > $SCRATCH/add_zero.pdll <<'EOF'
#include "mlir/Dialect/Arith/IR/ArithOps.td"

Constraint IsZero(attr: Attr) [{
  return success(cast<::mlir::IntegerAttr>(attr).getValue().isZero());
}];

Pattern EliminateAddZero {
  let root = op<arith.addi>(x: Value, op<arith.constant> {value = zeroAttr: Attr});
  IsZero(zeroAttr);
  replace root with x;
}
EOF
mlir-pdll -x=mlir -I $LLVM_INCLUDE $SCRATCH/add_zero.pdll

# Repetition means equality: match x - x, rewrite to constant zero.
# (Include line added per §3.1/3.2 -- every scratch file starts with the ODS include.)
cat > $SCRATCH/zero_self_sub.pdll <<'EOF'
#include "mlir/Dialect/Arith/IR/ArithOps.td"

Pattern ZeroSelfSub {
  let root = op<arith.subi>(x: Value, x);
  rewrite root with {
    let zero = op<arith.constant> {value = attr<"0 : i32">};
    replace root with zero;
  };
}
EOF
mlir-pdll -x=mlir -I $LLVM_INCLUDE $SCRATCH/zero_self_sub.pdll

{ step "3.8 Rewrites: the full menu"; } 2>/dev/null
# "Verified: a pattern calling Double(x) inside its rewrite block compiles,
# the helper inlined into the generated PDL."
cat > $SCRATCH/rewrites.pdll <<'EOF'
#include "mlir/Dialect/Arith/IR/ArithOps.td"

Rewrite Double(v: Value) -> Op {
  return op<arith.addi>(v, v);
}
Rewrite EraseOp(op: Op) => erase op;

Pattern DoubleViaRewriteHelper {
  let root = op<arith.muli>(x: Value, op<arith.constant>);
  rewrite root with {
    let doubled = Double(x);
    replace root with doubled;
  };
}
EOF
mlir-pdll -x=mlir -I $LLVM_INCLUDE $SCRATCH/rewrites.pdll

{ step "5. The pass: parsing patterns at runtime"; } 2>/dev/null
# "What did the generator actually emit? Look (verified, mlir-pdll -x=cpp)"
mlir-pdll -x=cpp -I $LLVM_INCLUDE ../../lib/Transform/Arith/MulToAdd.pdll

{ step "6. Under the hood: watching a pattern become bytecode"; } 2>/dev/null
# First, PDLL -> PDL -- section 2's command on the repo's file:
mlir-pdll -x=mlir -I $LLVM_INCLUDE ../../lib/Transform/Arith/MulToAdd.pdll

# Second, PDL -> pdl_interp -- run the standard lowering:
mlir-pdll -x=mlir -I $LLVM_INCLUDE ../../lib/Transform/Arith/MulToAdd.pdll \
  | mlir-opt --convert-pdl-to-pdl-interp

{ step "7. Build integration and running it"; } 2>/dev/null
$TUTORIAL_OPT --mul-to-add-pdll $TESTS/mul_to_add_pdll.mlir

# skipped: bazel test //tests:mul_to_add_pdll.mlir.test (build-system test runner, see §7)
# skipped: llvm-lit -sv build-ninja/tests --filter mul_to_add_pdll (CMake test runner; build-ninja is stale, see §7)

rm -rf $SCRATCH
set +x # turn off command echoing
