#!/bin/bash

# run relative to this script's location (tutorial/script/), so it works from any cwd
cd "$(dirname "$0")" || exit 1

# print a section banner. The { ...; } 2>/dev/null redirections keep set -x
# trace noise out of the output: inside the function for its own commands,
# and at each call site ({ step "..."; } 2>/dev/null) for the call itself.
step() { { set +x; } 2>/dev/null; echo; echo "### $* ###"; set -x; }

set -x  # turn on command echoing
TESTS="../../tests"

{ step "2. The example program (01-mlir-basics-and-running-a-lowering.md)"; } 2>/dev/null
cat $TESTS/ctlz_simple.mlir

{ step "3. Step: run mlir-opt with no passes"; } 2>/dev/null
mlir-opt $TESTS/ctlz_simple.mlir

mlir-opt -- $TESTS/ctlz_simple.mlir

# expected to FAIL: the verifier demo (i32 * i64 mismatch)
mlir-opt -- $TESTS/wrong_type.mlir

{ step "3. Step: run mlir-opt with --mlir-print-op-generic"; } 2>/dev/null
mlir-opt --mlir-print-op-generic $TESTS/ctlz_simple.mlir

{ step "4. Step: apply the lowering"; } 2>/dev/null
mlir-opt --convert-math-to-funcs=convert-ctlz $TESTS/ctlz_simple.mlir

set +x # turn off command echoing
