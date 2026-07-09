// A runnable demonstration of the FileCheck directive family, explained
// step by step in tutorial/02-testing-a-lowering.md section 3. Each RUN
// line exercises one group of check directives, selected with
// --check-prefix. Two of the groups are assertions that deliberately FAIL;
// their RUN lines wrap FileCheck in LLVM's `not` utility, which inverts
// the exit code, so this file as a whole passes as a test.

// RUN: mlir-opt %s | FileCheck %s --check-prefix=LOOSE
// RUN: mlir-opt %s | not FileCheck %s --check-prefix=STRICT
// RUN: mlir-opt %s | FileCheck %s --check-prefix=SIG
// RUN: mlir-opt %s | not FileCheck %s --check-prefix=SIGBAD
// RUN: mlir-opt %s | FileCheck %s --check-prefix=CAP

// Two functions; only the SECOND contains an arith.addi.
func.func @square(%arg0: i32) -> i32 {
  %0 = arith.muli %arg0, %arg0 : i32
  func.return %0 : i32
}

func.func @double(%arg0: i32) -> i32 {
  %0 = arith.addi %arg0, %arg0 : i32
  func.return %0 : i32
}

// The LOOSE group passes even though @square contains no arith.addi — a
// plain check may skip any number of lines, including the end of @square,
// so it finds the arith.addi inside @double.
// LOOSE: func.func @square
// LOOSE: arith.addi

// The STRICT group makes the same bogus assertion, but its labels confine
// each check to one function, so the bug is caught. (This group fails
// FileCheck; `not` in its RUN line inverts that.)
// STRICT-LABEL: func.func @square
// STRICT: arith.addi
// STRICT-LABEL: func.func @double

// The SIG group matches pieces of @square's signature; -SAME continues on
// the same output line as the previous match.
// SIG-LABEL: func.func @square(
// SIG-SAME: %arg0: i32
// SIG-SAME: -> i32

// The SIGBAD group demands arith.muli on the signature line; it is on the
// next line, so this fails (inverted by `not` in the RUN line).
// SIGBAD-LABEL: func.func @square(
// SIGBAD-SAME: arith.muli

// The CAP group uses capture variables — double brackets capture text and
// assert the same text reappears: both operands of the multiply are the
// same value, and the value returned is the multiply's result.
// CAP-LABEL: func.func @square(
// CAP: %[[V:.*]] = arith.muli %[[A:.*]], %[[A]] : i32
// CAP: return %[[V]] : i32
