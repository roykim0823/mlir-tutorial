# Tutorial 9: Canonicalizers and Declarative Rewrite Patterns

This is a step-by-step companion to the article
[Canonicalizers and Declarative Rewrite Patterns](https://jeremykun.com/2023/09/20/mlir-canonicalizers-and-declarative-rewrite-patterns/).
[Tutorial 7](07-folders-and-constant-propagation.md) gave `poly` folders —
single-op, constants-only simplifications. This tutorial adds the general
case: **canonicalization patterns**, rewrites spanning several ops, hooked
into the same `--canonicalize` pass. Along the way we learn a second
authoring language for rewrite patterns — **DRR** (declarative rewrite
rules), tablegen instead of C++ — and resolve the last unexplained lines
in the `poly` codebase: `let hasCanonicalizer = 1` and
`PolyPatterns.td`.

The running examples are two genuinely mathematical rewrites:

- **Difference of squares**: x² − y² = (x + y)(x − y) — trades two
  (expensive) multiplications for one, plus cheap additions. Tutorial 3's
  `MulToAdd` economics, at the polynomial level.
- **Conjugate through evaluation**: for polynomials with real
  coefficients, f(z̄) = f(z)̄ — evaluating at a conjugate equals
  conjugating the evaluation. Rewriting the former to the latter puts IR
  in a *normal form* so equivalent computations look identical.

**What you will learn:**

- What canonicalization is, and how it differs from folding.
- `hasCanonicalizer` / `getCanonicalizationPatterns` — attaching rewrite
  patterns (Tutorial 3's kind!) to an op's canonical form.
- The same pattern written twice: C++ (the article's version) and DRR
  (the repo's version) — and how to read the tablegen pattern language
  (`Pat`, `Pattern`, bound variables, constraints, `CPred`).
- How `-gen-rewriters` turns DRR into the C++ you would have written.
- When to choose folding vs. canonicalization vs. a standalone pass.

**Prerequisites:** [Tutorial 7](07-folders-and-constant-propagation.md)
(folding, `--canonicalize`) and [Tutorial 3](03-writing-our-first-pass.md)
(`OpRewritePattern`, `matchAndRewrite`). Same build setup
(`$TUTORIAL_OPT`).

---

## 1. Concepts: canonicalization vs. folding

Tutorial 7 drew folding's box deliberately small: one op, no new ops,
return an attribute or existing value. Plenty of valuable rewrites don't
fit:

- x² − y² → (x + y)(x − y) touches *three* ops (a sub and two muls) and
  creates three new ones.
- f(conj(z)) → conj(f(z)) *reorders* two ops.

These are **DAG-to-DAG rewrites** — replacing one connected piece of the
IR graph with another — and MLIR's home for them is the
**canonicalization** hook: patterns an op contributes to the
`--canonicalize` pass. That pass, which you've been using since
Tutorial 7, is really three things braided together, run by the greedy
driver (Tutorial 3 §7) until a fixed point:

1. every op's **folders**,
2. every op's **canonicalization patterns** (this tutorial),
3. generic cleanups (dead-op removal — the reason canonicalize deletes
   what sccp leaves behind).

Why "canonical"? The goal isn't only optimization: it's putting IR into a
*normal form* so that later passes (and folders, and CSE) see one shape
instead of five equivalent ones. The conjugation rewrite is pure
normal-form play: `conj` ops migrate to a consistent position, so
downstream code needs one pattern, not two. (The article notes, fairly,
that in practice canonicalize accretes minor optimizations too and
becomes "a heavyweight and powerful pass" — the boundary between
"canonical form" and "optimization" is community convention.)

Mechanically you already know everything involved: canonicalization
patterns *are* Tutorial 3's `RewritePattern`s. The only new machinery is
how an op volunteers them.

## 2. Attaching patterns to an op

In `PolyOps.td` (Tutorial 5), the binops and `eval` declare:

```tablegen
let hasCanonicalizer = 1;
```

which generates one static-method declaration on the op class:

```cpp
static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results,
                                        ::mlir::MLIRContext *context);
```

The `--canonicalize` pass calls this on every registered op type and
collects the patterns. The implementations at the bottom of
[`PolyOps.cpp`](../lib/Dialect/Poly/PolyOps.cpp) are one line each:

```cpp
void SubOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results,
                                        ::mlir::MLIRContext *context) {
  results.add<DifferenceOfSquares>(context);
}

void EvalOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results,
                                         ::mlir::MLIRContext *context) {
  results.add<LiftConjThroughEval>(context);
}
```

(`AddOp`'s and `MulOp`'s are registered but empty — hooks waiting for
future patterns.) A design question worth pausing on: each pattern is
attached to its *root* op — the op at the "top" of the matched DAG, the
one being replaced. Difference-of-squares roots at the `sub`; the
conjugation rewrite roots at the `eval`.

## 3. The pattern in C++ (the article's version)

The article first writes `DifferenceOfSquares` exactly the way Tutorial 3
taught — worth reading in full because the DRR version must be understood
*as shorthand for this*:

```cpp
struct DifferenceOfSquares : public OpRewritePattern<SubOp> {
  DifferenceOfSquares(mlir::MLIRContext *context)
      : OpRewritePattern<SubOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(SubOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getOperand(0);
    Value rhs = op.getOperand(1);
    if (!lhs.hasOneUse() || !rhs.hasOneUse()) {
      return failure();
    }

    auto rhsMul = rhs.getDefiningOp<MulOp>();
    auto lhsMul = lhs.getDefiningOp<MulOp>();
    if (!rhsMul || !lhsMul) {
      return failure();
    }

    bool rhsMulOpsAgree = rhsMul.getLhs() == rhsMul.getRhs();
    bool lhsMulOpsAgree = lhsMul.getLhs() == lhsMul.getRhs();
    if (!rhsMulOpsAgree || !lhsMulOpsAgree) {
      return failure();
    }

    auto x = lhsMul.getLhs();
    auto y = rhsMul.getLhs();

    AddOp newAdd = rewriter.create<AddOp>(op.getLoc(), x, y);
    SubOp newSub = rewriter.create<SubOp>(op.getLoc(), x, y);
    MulOp newMul = rewriter.create<MulOp>(op.getLoc(), newAdd, newSub);

    rewriter.replaceOp(op, {newMul});
    return success();
  }
};
```

All Tutorial 3 vocabulary — `getDefiningOp` to climb the SSA graph,
match-phase bailouts before any mutation, `create`/`replaceOp`. Two
details are new and semantic, not mechanical:

- **The `hasOneUse` guard.** If `x²` has *another* consumer, rewriting
  the sub doesn't retire the mul — it would still be computed, and we'd
  have *added* ops for nothing. "Is this rewrite profitable?" often
  reduces to use-count checks.
- **What's *not* here: erasing the muls.** The pattern replaces only the
  root `sub`; the old muls become dead and the canonicalizer's built-in
  dead-code cleanup (section 1, item 3) collects them. Rewrite the root,
  let the driver sweep.

> **Want to run this exact pattern?** It is kept buildable in
> [`tutorial/code/09/`](code/09/), wrapped as a standalone pass so the
> Poly dialect in `lib/` needn't be edited:
>
> ```bash
> bazel build //tutorial/code/09:tutorial-opt-09
> bazel-bin/tutorial/code/09/tutorial-opt-09 tests/poly_canonicalize.mlir --difference-of-squares
> ```
>
> On the section 5 test file it produces the same rewrite as the DRR
> version (and honors the same `hasOneUse` refusal). One line had to
> change to compile against current MLIR — see the directory's README.

## 4. The same pattern in DRR (the repo's version)

Look closely at section 3: about 30 lines, of which perhaps 6 carry the
*idea* — "match sub(mul(x,x), mul(y,y)), emit mul(add(x,y), sub(x,y))".
The rest is casts, null checks, and plumbing. **DRR** (Declarative
Rewrite Rules) is a tablegen language for exactly this shape of pattern:
say the source DAG, say the target DAG, let a generator write the
plumbing. The repo's current implementation, from
[`lib/Dialect/Poly/PolyPatterns.td`](../lib/Dialect/Poly/PolyPatterns.td):

```tablegen
def HasOneUse: Constraint<CPred<"$_self.hasOneUse()">, "has one use">;

// Rewrites (x^2 - y^2) as (x+y)(x-y) if x^2 and y^2 have no other uses.
def DifferenceOfSquares : Pattern<
  (Poly_SubOp (Poly_MulOp:$lhs $x, $x), (Poly_MulOp:$rhs $y, $y)),
  [
    (Poly_AddOp:$sum $x, $y),
    (Poly_SubOp:$diff $x, $y),
    (Poly_MulOp:$res $sum, $diff),
  ],
  [(HasOneUse:$lhs), (HasOneUse:$rhs)]
>;
```

Reading the language:

- **The source pattern** (first argument) is the DAG to match, written as
  nested op applications: a `Poly_SubOp` whose operands are two
  `Poly_MulOp`s. Names with `$` are *bound variables*: `$x` appearing
  twice in `(Poly_MulOp:$lhs $x, $x)` means "both operands of this mul
  must be the *same* value" — the C++ version's `getLhs() == getRhs()`
  check, expressed by repetition. `OpName:$name` binds the matched *op's
  result* itself (`$lhs`, `$rhs`) for use in constraints.
- **The result pattern(s)** (second argument) build the replacement.
  Because this replacement is a *list* of new ops (add, sub, mul), the
  `Pattern` class is used; the last entry (`$res`) replaces the root op's
  result. A single-op replacement can use the simpler `Pat<source,
  result>` form — see the next pattern.
- **Constraints** (third argument) are extra predicates. There is no
  built-in DRR way to say "has one use", so `HasOneUse` *injects C++*
  via `CPred`: `$_self` expands to the constrained value, and the string
  is pasted into the generated match code. DRR's escape hatch is
  literally "write the C++ inline" — the author's candid note is that
  fluency in *both* styles is required, since real patterns keep one foot
  in C++.

The conjugation rewrite is the simple form — one op in, one op out,
`Pat`:

```tablegen
def LiftConjThroughEval : Pat<
  (Poly_EvalOp $f, (ConjOp $z, $fastmath)),
  (ConjOp (Poly_EvalOp $f, $z), $fastmath)
>;
```

Match "eval of f at conj(z)", emit "conj of eval of f at z" — the
math identity f(z̄) = f(z)̄, as two lines of tree surgery. (`ConjOp` comes
from including `mlir/Dialect/Complex/IR/ComplexOps.td` — DRR patterns mix
dialects freely. `$fastmath` binds `conj`'s fastmath *attribute* and
carries it to the new op: attributes bind just like operands.)

### From DRR to C++

The build runs the `-gen-rewriters` tablegen backend over
`PolyPatterns.td` (the `canonicalize_inc_gen` target in the
[`BUILD`](../lib/Dialect/Poly/BUILD) file, Tutorial 5 §6's fourth
`gentbl_cc_library`), producing `PolyCanonicalize.cpp.inc`, which
`PolyOps.cpp` includes. Run it yourself and skim (Tutorial 4's habit):

```bash
mlir-tblgen --gen-rewriters -I /opt/homebrew/opt/llvm@20/include \
  -I lib/Dialect/Poly lib/Dialect/Poly/PolyPatterns.td
```

The output is (verified) a `RewritePattern` subclass per `def`, a
`matchAndRewrite` that walks defining ops and checks bound-variable
equality, your `CPred` strings pasted in —

```cpp
struct DifferenceOfSquares : public ::mlir::RewritePattern {
  ...
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0, ...) {
    ...
    if (!(((*lhs.getODSResults(0).begin()).hasOneUse()))) { ... }
```

— plus a convenience `populateWithGenerated(RewritePatternSet&)` that
registers everything at once (this repo registers by name instead, in
section 2's hooks, since different patterns belong to different ops). As
with every tablegen backend since Tutorial 4: it's a see-through
generator, and when a DRR pattern misbehaves, you debug by reading this
file.

## 5. Step: watch the canonicalizations

[`tests/poly_canonicalize.mlir`](../tests/poly_canonicalize.mlir) has
four functions; take them in turn with
`$TUTORIAL_OPT --canonicalize tests/poly_canonicalize.mlir` (all output
below is real).

**Difference of squares, firing:**

```mlir
func.func @test_difference_of_squares(
    %0: !poly.poly<3>, %1: !poly.poly<3>) -> !poly.poly<3> {
  %2 = poly.mul %0, %0 : !poly.poly<3>
  %3 = poly.mul %1, %1 : !poly.poly<3>
  %4 = poly.sub %2, %3 : !poly.poly<3>
  %5 = poly.add %4, %4 : !poly.poly<3>
  return %5 : !poly.poly<3>
}
```

becomes

```mlir
func.func @test_difference_of_squares(%arg0: !poly.poly<3>, %arg1: !poly.poly<3>) -> !poly.poly<3> {
  %0 = poly.add %arg0, %arg1 : !poly.poly<3>
  %1 = poly.sub %arg0, %arg1 : !poly.poly<3>
  %2 = poly.mul %0, %1 : !poly.poly<3>
  %3 = poly.add %2, %2 : !poly.poly<3>
  return %3 : !poly.poly<3>
}
```

Two muls in, one mul out — and notice the original muls are simply
*gone*: the dead-code sweep from section 3's last bullet, observed.

**Difference of squares, declining.** The third test function adds one
line — `%5 = poly.add %4, %2` — giving `x²` a second use. Run it: the IR
comes back *unchanged*. The `HasOneUse` constraint working as designed;
a rewrite that fires only when profitable.

**Conjugate through eval:**

```mlir
%z_bar = complex.conj %z : complex<f64>
%evaled = poly.eval %f, %z_bar : (!poly.poly<3>, complex<f64>) -> complex<f64>
```

becomes

```mlir
%0 = poly.eval %arg0, %arg1 : (!poly.poly<3>, complex<f64>) -> complex<f64>
%1 = complex.conj %0 : complex<f64>
```

The conjugation lifted through the evaluation — and this is why
Tutorial 8's verifier admits complex points at all.

**And the first function** is Tutorial 7's fold demo (constants all the
way down to a `poly.constant dense<[2, 4, 6]>`) — a reminder that
`--canonicalize` is folders *and* patterns *and* cleanup in one loop.

Run the suite entry:

```bash
bazel test //tests:poly_canonicalize.mlir.test              # Bazel
llvm-lit -sv build-ninja/tests --filter poly_canonicalize   # CMake
```

## 6. Choosing your tool, final table

Three tutorials of IR-simplification machinery, one decision rule:

| Mechanism | Scope | Creates ops? | Runs |
|---|---|---|---|
| **Folder** (Tut. 7) | one op | never (attribute/value out) | constantly: greedy driver, canonicalize, sccp, after every `create` in some drivers |
| **Canonicalization pattern** (here) | a DAG | freely | inside `--canonicalize` |
| **Standalone pass** (Tut. 3) | whole module/function | freely | when *you* schedule it |

Prefer the smallest box that fits: folders if it's one op and constants;
canonicalization if it's a local identity that should *always* hold
(cheap, universally beneficial, direction-agreed); a dedicated pass if
it's a strategy — profitable only sometimes, needing analysis or
configuration (Tutorial 3's full-unroll is a pass, not a
canonicalization, because unrolling everything is not always an
improvement).

## Differences from the original article

- **`DifferenceOfSquares` migrated from C++ to DRR.** The article writes
  it as section 3's C++ class and leaves it that way; the repo's current
  code ships the DRR version of section 4, registered under the same
  name. (Both are shown above deliberately — the article's C++ is the
  best possible reading guide to the repo's tablegen.)
- **`ConjOp` grew a `fastmath` attribute** upstream, so the repo's
  `LiftConjThroughEval` binds and forwards `$fastmath`; the article's
  two-argument version no longer compiles.
- The article mentions PDLL as a third pattern language it hasn't
  explored; the repo has since acquired a PDLL example
  (`--mul-to-add-pdll`, `lib/Transform/Arith/MulToAdd.pdll`) — that's the
  final article/tutorial of the series.

## Where to go next

The `poly` dialect is now feature-complete as a *high-level* IR: syntax,
semantics, verification, folding, canonicalization. What it cannot yet do
is *run*. The remaining arc is downhill — Tutorial 1's lowering
staircase, but for our own dialect: convert `poly` ops into `arith` and
`tensor` ops
([Tutorial 10: Dialect Conversion](10-dialect-conversion.md)),
then to LLVM and an actual executable (Tutorial 11). The
`lib/Conversion/PolyToStandard/` directory has been waiting since
Tutorial 3's project-layout tour.

**Exercises**

1. Reproduce section 5, then break the `HasOneUse` guard: delete the
   constraint list from `DifferenceOfSquares` in a *copy* of
   `PolyPatterns.td`, regenerate with `-gen-rewriters`, and read the
   generated `matchAndRewrite` diff. (No rebuild needed — this is a
   tablegen-only exercise.)
2. The conjugation identity f(z̄) = f(z)̄ requires *real* coefficients —
   ours are integers, so it holds. Write the one-line DRR `Pat` for the
   related identity eval(f, conj(conj(z))) → eval(f, z)... then explain
   why you don't need to: which existing upstream canonicalization
   already handles conj∘conj? (Check: run two nested `complex.conj`
   through `--canonicalize`.)
3. Write x² − y² with `x = y` (both operands the same value) and
   canonicalize. The pattern fires and leaves `poly.sub %x, %x` in the
   output (verified) — semantically zero, but nothing simplifies it,
   because Tutorial 7's `SubOp::fold` only handles *constant* operands.
   Sketch the fold (or DRR pattern) that would finish the job: `x − x →`
   what, exactly? (Careful: the answer is a *constant zero polynomial* —
   which existing op materializes it, and what attribute does its folder
   need to return?)
4. In Tutorial 3, `PowerOfTwoExpand` and `PeelFromMul` were *pass*
   patterns, not canonicalizations. Given section 6's table, argue both
   sides: what would go wrong (or right) if `mul-to-add` rewrites were
   attached to `arith.muli`'s canonicalizer?
5. DRR can't express "has one use" natively, but it *can* express
   attribute equality and type constraints. Skim the DRR docs
   (mlir.llvm.org/docs/DeclarativeRewrites/) and find one feature not
   used in `PolyPatterns.td` (e.g. `NativeCodeCall`) — sketch a use for
   it in `poly`.
