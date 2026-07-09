# Tutorial 9 code: the C++ pattern, before DRR

[`DifferenceOfSquares.cpp`](DifferenceOfSquares.cpp) contains the C++
`DifferenceOfSquares` rewrite pattern exactly as
[Tutorial 9 §3](../../09-canonicalizers-and-drr.md) presents it — the
article's original version, which the repo has since replaced with the DRR
pattern in `lib/Dialect/Poly/PolyPatterns.td` (Tutorial 9 §4).

Two pieces of scaffolding around the pattern are **not** from the article:

- The article registered the pattern in
  `SubOp::getCanonicalizationPatterns`, so it ran inside `--canonicalize`.
  Here it is wrapped in a standalone `PassWrapper` pass (flag:
  `--difference-of-squares`) so you can run it without editing the Poly
  dialect in `lib/`.
- One line differs from the article listing: `replaceOp(op, {newMul})`
  became `replaceOp(op, newMul)` — the braced form is ambiguous
  (`ValueRange` vs. `Operation*`) against current MLIR and no longer
  compiles. A source comment marks the spot.

Build and run (verified):

```bash
bazel build //tutorial/code/09:tutorial-opt-09
bazel-bin/tutorial/code/09/tutorial-opt-09 tests/poly_canonicalize.mlir --difference-of-squares
```

On that test file, `@test_difference_of_squares` is rewritten to
`poly.add` / `poly.sub` / `poly.mul` — the same result as the DRR version
under `--canonicalize` — and `@test_difference_of_squares_other_uses` is
left untouched (the `hasOneUse` guard declines). The conj-through-eval
function is also untouched: that is `EvalOp`'s canonicalization pattern,
which this pass deliberately does not include. `@test_simple` still gets
constant-folded, because the greedy driver runs folders even when no
pattern matches — Tutorial 3 §7's "AndFold" behavior.

> **CMake users:** `cmake --build build --target tutorial-opt-09`, subject
> to the LLVM-submodule requirement (see the top-level README).
