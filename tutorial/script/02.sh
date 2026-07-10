#!/bin/bash

# run relative to this script's location (tutorial/script/), so it works from any cwd
cd "$(dirname "$0")" || exit 1

# print a section banner. The { ...; } 2>/dev/null redirections keep set -x
# trace noise out of the output: inside the function for its own commands,
# and at each call site ({ step "..."; } 2>/dev/null) for the call itself.
step() { { set +x; } 2>/dev/null; echo; echo "### $* ###"; set -x; }

set -x  # turn on command echoing
TESTS="../../tests"

{ step "3. The example program (02-testing-a-lowering.md)"; } 2>/dev/null
cat $TESTS/filecheck_directives.mlir

mlir-opt $TESTS/filecheck_directives.mlir | FileCheck $TESTS/filecheck_directives.mlir --check-prefix=LOOSE
echo $?

mlir-opt $TESTS/filecheck_directives.mlir | FileCheck $TESTS/filecheck_directives.mlir --check-prefix=STRICT
echo $?

mlir-opt $TESTS/filecheck_directives.mlir | FileCheck $TESTS/filecheck_directives.mlir --check-prefix=SIG
echo $?

mlir-opt $TESTS/filecheck_directives.mlir | FileCheck $TESTS/filecheck_directives.mlir --check-prefix=SIGBAD
echo $?

mlir-opt $TESTS/filecheck_directives.mlir | FileCheck $TESTS/filecheck_directives.mlir --check-prefix=CAP
echo $?

mlir-opt $TESTS/ctlz_simple.mlir | FileCheck $TESTS/ctlz_simple.mlir 

{ step "4. Step"; } 2>/dev/null
cat $TESTS/ctlz_simple.mlir

mlir-opt --convert-math-to-funcs=convert-ctlz $TESTS/ctlz_simple.mlir \
								 | FileCheck $TESTS/ctlz_simple.mlir

mlir-opt $TESTS/ctlz_simple.mlir | FileCheck $TESTS/ctlz_simple.mlir

{ step "5. Step: exhaustive asserions"; } 2>/dev/null
# (input must be ctlz.mlir itself — its CHECK lines were generated from
#  ctlz.mlir's @main; ctlz_simple.mlir's @main has a different signature)
mlir-opt --convert-math-to-funcs=convert-ctlz $TESTS/ctlz.mlir \
								 | FileCheck $TESTS/ctlz.mlir


curl -sLo /tmp/generate-test-checks.py \
  https://raw.githubusercontent.com/llvm/llvm-project/release/20.x/mlir/utils/generate-test-checks.py

mlir-opt --convert-math-to-funcs=convert-ctlz $TESTS/ctlz_simple.mlir \
								 | python3 /tmp/generate-test-checks.py

{ step "8. Bonus step"; } 2>/dev/null
mlir-opt $TESTS/ctlz_runner.mlir \
  -pass-pipeline="builtin.module( \
     convert-math-to-funcs{convert-ctlz}, \
     func.func(convert-scf-to-cf,convert-arith-to-llvm), \
     convert-func-to-llvm, \
     convert-cf-to-llvm, \
     reconcile-unrealized-casts)" \
  | mlir-runner -e test_7i32_to_29 -entry-point-result=i32

set +x # turn off command echoing
