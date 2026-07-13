# Chapter 3 code: hand-written passes, before tablegen

This is the code of
[Chapter 3: Writing Our First Pass](../../03-writing-our-first-pass.md)
exactly as the chapter presents it: every pass
is a hand-written `PassWrapper` subclass anchored on `func.func`, with its
CLI flag name in `getArgument()` and an explicit
`mlir::PassRegistration<...>()` line in the driver. **No tablegen is
involved anywhere.** The main tree's versions of these same passes
(`lib/Transform/Affine`, `lib/Transform/Arith`) use tablegen-generated
base classes instead — migrating from this code to that one is the subject
of [Chapter 4](../../04-using-tablegen-for-passes.md).

| File | Chapter section |
|---|---|
| `AffineFullUnroll.h` | §4 (the class, verbatim) |
| `AffineFullUnroll.cpp` | §5 (the walk-based `runOnOperation`) |
| `AffineFullUnrollPatternRewrite.h/.cpp` | §7 (pattern + driver invocation) |
| `MulToAdd.h/.cpp` | §8 (`PowerOfTwoExpand`, `PeelFromMul`) |
| `tutorial-opt.cpp` | §3 (the pre-tablegen driver with `PassRegistration` lines) |

Because these passes are anchored on `OperationPass<mlir::func::FuncOp>`,
`getOperation()` returns a `FuncOp` and the walk is written
`getOperation().walk` (dot) — the main tree's generic passes write
`getOperation()->walk` (arrow). Chapter 3 §5's "What the repo has now"
note explains this.

Deliberate deviations from the original hand-written files (this snapshot
follows the *chapter*, which follows the current repo, where they
disagree): include paths say `tutorial/code/03/...` instead of
`lib/Transform/...`; a `multiplicatoins` typo in the original
`MulToAdd` description is fixed; each file opens with an orientation
comment (added in this copy) saying what it is; and the pattern bodies
match the current `lib/` code apart from small clarifying comments added
in this copy.

Build and run:

```bash
# Bazel
bazel build //tutorial/code/03:tutorial-opt-03
bazel-bin/tutorial/code/03/tutorial-opt-03 tests/affine_loop_unroll.mlir --affine-full-unroll
bazel-bin/tutorial/code/03/tutorial-opt-03 tests/mul_to_add.mlir --mul-to-add
```

> **CMake users:** `cmake --build build --target tutorial-opt-03`, then run
> `build/tutorial/code/03/tutorial-opt-03` the same way. Requires the LLVM
> submodule build, like `tutorial-opt`.

Verified: on `tests/affine_loop_unroll.mlir` and `tests/mul_to_add.mlir`,
the binary's output is byte-identical to `tutorial-opt`'s for all three
passes (checked with `diff`), and it passes the same FileCheck assertions.
Its `--help` lists the three flags with the same descriptions — and *only*
those three passes: like the original, this driver has no
`registerAllPasses()`, so the hundreds of upstream passes (and
`--mul-to-add-pdll`, Chapter 13's code) that today's `tutorial-opt`
exposes are absent.
