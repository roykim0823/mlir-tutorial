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

{ step "3. Common subexpression elimination (06-using-traits.md)"; } 2>/dev/null
# before: the same product computed twice
$TUTORIAL_OPT $TESTS/cse.mlir
# after: one poly.mul remains, feeding both sides of the add
$TUTORIAL_OPT -cse $TESTS/cse.mlir

{ step "3. Loop-invariant code motion"; } 2>/dev/null
# before: a poly.mul of two loop-constant polynomials sits inside the loop
$TUTORIAL_OPT $TESTS/code_motion.mlir
# after: the mul is hoisted above the loop; the poly.add stays (it consumes
# %sum_iter, which changes every iteration)
$TUTORIAL_OPT --loop-invariant-code-motion $TESTS/code_motion.mlir

{ step "3. Control-flow sinking"; } 2>/dev/null
# before: two polynomials built ahead of the scf.if, each used in only one arm
$TUTORIAL_OPT $TESTS/control_flow_sink.mlir
# after: each poly.from_tensor (and its arith.constant) sinks into the arm
# that uses it
$TUTORIAL_OPT -control-flow-sink $TESTS/control_flow_sink.mlir

# Variant from §3: an op used by BOTH arms does not sink -- the pass moves
# ops, never clones them, so the output is identical to the input.
SCRATCH=$(mktemp -d)
cat > $SCRATCH/sink_shared.mlir <<'EOF'
func.func @test_shared_use(%arg0: i1) -> !poly.poly<10> {
  %0 = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
  %p0 = poly.from_tensor %0 : tensor<3xi32> -> !poly.poly<10>
  %4 = scf.if %arg0 -> (!poly.poly<10>) {
    %2 = poly.mul %p0, %p0 : !poly.poly<10>
    scf.yield %2 : !poly.poly<10>
  } else {
    %3 = poly.add %p0, %p0 : !poly.poly<10>
    scf.yield %3 : !poly.poly<10>
  }
  return %4 : !poly.poly<10>
}
EOF
# before (round-trip; set -x does not echo heredoc contents):
$TUTORIAL_OPT $SCRATCH/sink_shared.mlir
# after: identical -- %p0's uses span both arms, so nothing moves
$TUTORIAL_OPT -control-flow-sink $SCRATCH/sink_shared.mlir

# skipped: bazel test //tests:cse.mlir.test //tests:code_motion.mlir.test //tests:control_flow_sink.mlir.test (bazel test; see §3)
# skipped: llvm-lit -sv build-ninja/tests --filter 'cse|code_motion|control_flow_sink' (llvm-lit; build-ninja is stale; see §3)

{ step "3. The dog that didn't bark"; } 2>/dev/null
# poly.eval has no Pure trait, so CSE must not deduplicate it: both evals survive.
cat > $SCRATCH/dup_eval.mlir <<'EOF'
func.func @dup_eval(%p: !poly.poly<10>, %x: i32) -> i32 {
  %0 = poly.eval %p, %x : (!poly.poly<10>, i32) -> i32
  %1 = poly.eval %p, %x : (!poly.poly<10>, i32) -> i32
  %2 = arith.addi %0, %1 : i32
  return %2 : i32
}
EOF
# before (round-trip):
$TUTORIAL_OPT $SCRATCH/dup_eval.mlir
# after -cse: identical -- both evals survive, nobody told MLIR eval is pure
$TUTORIAL_OPT -cse $SCRATCH/dup_eval.mlir

{ step "3. The rest of the pass list"; } 2>/dev/null
# Passes that need nothing from poly: no memory ops, no dead values, no dead
# symbols -- stacked together the file comes out unchanged (identical to the
# cse.mlir "before" round-trip in the CSE demo above).
$TUTORIAL_OPT -mem2reg -sroa -remove-dead-values -symbol-dce $TESTS/cse.mlir

# -inline crashes THIS tool: func promises DialectInlinerInterface, but
# tutorial-opt never registers the extension implementing it. (Failure demo;
# stack trace trimmed to the error message.)
$TUTORIAL_OPT -inline $TESTS/cse.mlir 2>&1 | head -4

# The same inlining works under stock mlir-opt, which registers the extension.
cat > $SCRATCH/inline_demo.mlir <<'EOF'
func.func private @callee(%x: i32) -> i32 {
  %0 = arith.addi %x, %x : i32
  return %0 : i32
}
func.func @caller(%x: i32) -> i32 {
  %0 = call @callee(%x) : (i32) -> i32
  return %0 : i32
}
EOF
# before (round-trip): caller and callee both present
mlir-opt $SCRATCH/inline_demo.mlir
# after: callee's body inlined into caller (the private callee then dies)
mlir-opt -inline $SCRATCH/inline_demo.mlir

# -sccp needs folders (Chapter 7); the repo ops already have them.
# before: poly.mul/poly.add chains on constants
$TUTORIAL_OPT $TESTS/sccp.mlir
# after: the constant poly arithmetic folds to poly.constant
$TUTORIAL_OPT -sccp $TESTS/sccp.mlir

rm -rf $SCRATCH

set +x # turn off command echoing
