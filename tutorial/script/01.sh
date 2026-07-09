#!/bin/bash

# run relative to this script's location (tutorial/script/), so it works from any cwd
cd "$(dirname "$0")" || exit 1

set -x  # turn on command echoing
TESTS="../../tests"

# 2. The example program
cat $TESTS/ctlz_simple.mlir 

# 3. Step
mlir-opt $TESTS/ctlz_simple.mlir 

mlir-opt -- $TESTS/ctlz_simple.mlir

mlir-opt -- $TESTS/wrong_type.mlir 

mlir-opt --convert-math-to-funcs=convert-ctlz $TESTS/ctlz_simple.mlir 

set +x # turn off command echoing
