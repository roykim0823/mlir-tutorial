# Tutorial 3 code: hand-written passes, before tablegen

This is the code of
[Tutorial 3: Writing Our First Pass](../../03-writing-our-first-pass.md)
exactly as the tutorial (and the original article) presents it: every pass
is a hand-written `PassWrapper` subclass anchored on `func.func`, with its
CLI flag name in `getArgument()` and an explicit
`mlir::PassRegistration<...>()` line in the driver. **No tablegen is
involved anywhere.** The main tree's versions of these same passes
(`lib/Transform/Affine`, `lib/Transform/Arith`) use tablegen-generated
base classes instead — migrating from this code to that one is the subject
of [Tutorial 4](../../04-using-tablegen-for-passes.md).

| File | Tutorial section |
|---|---|
| `AffineFullUnroll.h` | §4 (the class, verbatim) |
| `AffineFullUnroll.cpp` | §5 (the walk-based `runOnOperation`) |
| `AffineFullUnrollPatternRewrite.h/.cpp` | §7 (pattern + driver invocation) |
| `MulToAdd.h/.cpp` | §8 (`PowerOfTwoExpand`, `PeelFromMul`) |
| `tutorial-opt.cpp` | §3 (the article-era driver with `PassRegistration` lines) |

Because these passes are anchored on `OperationPass<mlir::func::FuncOp>`,
`getOperation()` returns a `FuncOp` and the walk is written
`getOperation().walk` (dot) — the main tree's generic passes write
`getOperation()->walk` (arrow). Tutorial 3's "Differences from the
original article" section explains this.

Deliberate deviations from the article's original files (this snapshot
follows the *tutorial*, which follows the current repo, where they
disagree): include paths say `tutorial/code/03/...` instead of
`lib/Transform/...`; a `multiplicatoins` typo in the article-era
`MulToAdd` description is fixed; the pattern bodies match the current
`lib/` code character for character.

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
Its `--help` lists the three flags with the same descriptions (plus the
hundreds of upstream passes, but minus `--mul-to-add-pdll`, which belongs
to Tutorial 13's code).
