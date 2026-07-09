# Tutorial 3: Writing Our First Pass

This is a step-by-step companion to the article
[Writing Our First Pass](https://jeremykun.com/2023/08/10/mlir-writing-our-first-pass/).
It picks up where [Tutorial 2](02-testing-a-lowering.md) left off: we can run
upstream passes and test them — now we write passes of our own, organize them
into a project, and build `tutorial-opt`, this project's own version of
`mlir-opt` that carries them. The repo has evolved since the article was
written, so some code differs from the article's listings (see
[Differences from the original article](#differences-from-the-original-article)).

**What you will learn:**

- How an MLIR project is organized on disk, and why an out-of-tree project
  needs its own `opt` tool.
- The anatomy of a pass: `runOnOperation`, pass *anchoring*, and why
  anchoring exists (hint: parallelism).
- Two ways to implement a pass: walking the IR directly, and using the
  **pattern rewrite engine**.
- The core `PatternRewriter` API: matching, creating, replacing, and
  erasing operations.
- How the greedy rewrite driver applies competing patterns to a fixed
  point — with folding thrown in for free.

**Prerequisites:** Tutorials [1](01-mlir-basics-and-running-a-lowering.md)
and [2](02-testing-a-lowering.md), and a build that can produce
`tutorial-opt`. With Bazel that is automatic (the commands below build it on
first use). With CMake, `tutorial-opt` only builds against the LLVM
**submodule** — you must have checked out and built
`externals/llvm-project` per the README; a system-installed LLVM will not
work (see Tutorial 2's CMake notes).

Set a shell variable once so the commands below work for either build
system, mirroring Tutorial 2's pattern:

```bash
# Bazel (build it once, then use the binary directly):
bazel build //tools:tutorial-opt
TUTORIAL_OPT=bazel-bin/tools/tutorial-opt

# CMake:
cmake --build build-ninja --target tutorial-opt
TUTORIAL_OPT=build-ninja/tools/tutorial-opt
```

---

## 1. Concepts: passes, and why loop unrolling is the example

A **pass** is a unit of work that traverses the IR and analyzes or
transforms it. Everything a compiler does between parsing and emitting
machine code is organized as a pipeline of passes — the lowerings of
Tutorials 1–2 were passes; so are classic optimizations. A pass doesn't
have to lower anything: it can rewrite IR *within* one abstraction level,
which is what both passes in this tutorial do.

Our first example operates on the **`affine` dialect** — the highest-level
loop dialect in MLIR, and the reason Tutorial 1 said "keep the program at
the right abstraction level for each optimization." Affine loops are loops
whose bounds and index expressions are *affine functions* (sums of
variables times constants, like `2*i + j + 5`) of loop variables and
symbols. That restriction is what the dialect sells: it was originally
built for polyhedral loop analysis — a mathematical framework that studies
loop nests via lattice theory — and it makes questions like "what is this
loop's trip count?" statically answerable. An `scf.for` loop can have any
dynamic bound; an `affine.for` loop is analyzable by construction.

We exploit exactly that: our pass **fully unrolls** every affine loop —
replacing the loop with one copy of its body per iteration. Unrolling is a
real optimization (it removes branch overhead and exposes parallelism, and
in some domains — like the FHE compilers this series builds toward — loops
must be unrolled entirely because the target can't branch at all). But
honestly, we picked it because MLIR provides a one-call utility,
`loopUnrollFull`, that does the work. Both passes in this tutorial are
deliberately thin: the point is to learn how to *set up* a pass, navigate
the IR through the C++ API, and modify it — not to write a clever
algorithm.

## 2. Project organization

Before writing code, decide where it goes. A typical MLIR codebase (the
LLVM monorepo included) splits code into two parallel directory trees:

- `include/` — header files and tablegen files, defining the public
  interface;
- `lib/` — the implementation code.

Within each, conventional subdirectories signal a component's role:

- `Transform/` — passes that transform code *within* a dialect (this
  tutorial);
- `Conversion/` — passes that convert *between* dialects (the lowerings of
  Tutorial 1; this repo's `lib/Conversion/PolyToStandard` arrives in a
  later tutorial);
- `Analysis/` — passes that compute information without modifying IR;
- `Dialect/` — definitions of the dialects themselves.

This repo deviates in one deliberate way: **`include/` and `lib/` are
merged** — every header sits next to its implementation file. The parallel
tree exists in ordinary C++ projects partly to separate public interface
from private implementation, but Bazel has finer-grained ways to enforce
visibility, navigating mirrored trees is tedious, and this is a tutorial —
simpler is better. So the layout relevant to this tutorial is:

```
lib/Transform/Affine/
├── AffineFullUnroll.h                    # walk-based pass (this tutorial)
├── AffineFullUnroll.cpp
├── AffineFullUnrollPatternRewrite.h      # pattern-based pass (this tutorial)
├── AffineFullUnrollPatternRewrite.cpp
├── Passes.td, Passes.h                   # tablegen boilerplate (Tutorial 4)
├── BUILD                                 # Bazel targets
└── CMakeLists.txt
lib/Transform/Arith/
├── MulToAdd.h, MulToAdd.cpp              # this tutorial
├── ...                                   # PDLL variant, from a later article
tools/
├── tutorial-opt.cpp                      # the project's `opt` binary
├── BUILD
└── CMakeLists.txt
tests/
├── affine_loop_unroll.mlir
└── mul_to_add.mlir
```

The `tools/` convention: binaries live apart from libraries, and each
binary is a thin shell over library code — `tutorial-opt.cpp` is ~80 lines.

## 3. `tutorial-opt`: an `opt` tool of our own

Why can't we keep using `mlir-opt`? Because a binary only contains the
passes *compiled into it*. `mlir-opt` is built inside the LLVM project from
upstream code only; it has never heard of `affine-full-unroll`. In an
out-of-tree project (one living outside the LLVM monorepo, like this one),
you build your own tool.

MLIR makes this nearly free. It provides *registration hooks* to plug in
dialects and passes, and a library entry point, `MlirOptMain`, that
provides the entire `mlir-opt` command-line experience — parsing input,
verifying it, exposing every registered pass as a `--flag`, `--help`,
`--pass-pipeline`, IR printing options — on top of whatever you registered.
Here is [`tools/tutorial-opt.cpp`](../tools/tutorial-opt.cpp) trimmed to
the parts that matter for this tutorial (the full file also registers
dialects and a pipeline from later tutorials):

```cpp
#include "lib/Transform/Affine/Passes.h"
#include "lib/Transform/Arith/Passes.h"
#include "mlir/include/mlir/InitAllDialects.h"
#include "mlir/include/mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);

  mlir::tutorial::registerAffinePasses();
  mlir::tutorial::registerArithPasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Tutorial Pass Driver", registry));
}
```

Line by line:

- `DialectRegistry` collects the dialects the tool is allowed to *parse*.
  `registerAllDialects` throws in every upstream dialect — convenient for a
  tutorial; a production tool would register only what it handles, to keep
  the binary small.
- `registerAffinePasses()` / `registerArithPasses()` are generated
  functions (one per pass group; more below) that make our passes visible
  to the pass registry — this is what turns them into `--affine-full-unroll`
  style flags.
- `MlirOptMain(...)` runs the whole show; `asMainReturnCode` converts its
  `LogicalResult` into a process exit code.

In the original article, with no generated registration functions yet, the
same hookup was a single explicit line per pass:

```cpp
mlir::PassRegistration<mlir::tutorial::AffineFullUnrollPass>();
```

Either way, the effect is visible immediately:

```bash
$TUTORIAL_OPT --help | grep -E 'affine-full-unroll|mul-to-add'
```

```
      --affine-full-unroll         -   Fully unroll all affine loops
      --affine-full-unroll-rewrite -   Fully unroll all affine loops using the pattern rewrite engine
      --mul-to-add                 -   Convert multiplications to repeated additions
      --mul-to-add-pdll            -   Convert multiplications to repeated additions using pdll
```

Three of these are this tutorial's subject (`--mul-to-add-pdll` belongs to
a much later article). `--help` also lists hundreds of upstream passes:
your tool is a superset of `mlir-opt`.

## 4. Anatomy of a pass

The article defines the pass the from-scratch way, and it's worth
understanding in full even though the repo has since moved to generated
boilerplate (we'll get to that):

```cpp
class AffineFullUnrollPass
    : public PassWrapper<AffineFullUnrollPass,
                         OperationPass<mlir::func::FuncOp>> {
private:
  void runOnOperation() override;

  StringRef getArgument() const final { return "affine-full-unroll"; }

  StringRef getDescription() const final {
    return "Fully unroll all affine loops";
  }
};
```

- `runOnOperation()` is the heart of every pass — the method the pass
  manager calls to do the work.
- `getArgument()` is the CLI flag name; `getDescription()` is its `--help`
  text.
- `PassWrapper` fills in other required pass API (such as a compliant
  clone/copy method) using the *Curiously Recurring Template Pattern* —
  that's why the class passes itself as the first template argument. CRTP
  lets the base class call methods of the derived class without virtual
  dispatch; if it looks odd, treat it as an idiom to copy, not master.

The second template argument deserves its own paragraph.
`OperationPass<mlir::func::FuncOp>` **anchors** the pass to function
operations: the pass manager will invoke `runOnOperation()` once per
`func.func` in the module, and inside the pass, `getOperation()` returns
the `FuncOp` currently being processed.

Why anchor at all? **Parallelism.** MLIR's pass manager runs a
function-anchored pass on all functions *concurrently*. That is only safe
under two conditions, and the pass infrastructure enforces both:

1. The pass must not inspect or modify IR *outside* the operation it was
   given — otherwise two threads would race on shared IR. (This is why
   passes that need whole-program views anchor on the module instead, at
   the price of no parallelism.)
2. The anchor op itself must guarantee that nothing *inside* it can reach
   *outside*. `func.func` has this property (MLIR calls it
   `IsolatedFromAbove`): a function body cannot touch SSA values defined
   outside the function — which is a fancy way of saying functions can't
   screw with variables outside their own lexical scope.

That second condition explains why we can't just anchor the pass on
`affine.for` and skip the walk below: a loop body *can* affect the world
outside the loop (storing to memory defined elsewhere, yielding values), so
`affine.for` is not an isolation boundary, and MLIR won't let you build a
parallel pass on it.

### What the repo has now

The current
[`lib/Transform/Affine/AffineFullUnroll.cpp`](../lib/Transform/Affine/AffineFullUnroll.cpp)
expresses the same pass with generated boilerplate:

```cpp
#define GEN_PASS_DEF_AFFINEFULLUNROLL
#include "lib/Transform/Affine/Passes.h.inc"

struct AffineFullUnroll : impl::AffineFullUnrollBase<AffineFullUnroll> {
  using AffineFullUnrollBase::AffineFullUnrollBase;

  void runOnOperation() { /* section 5 */ }
};
```

The `impl::AffineFullUnrollBase` class — flag name, description,
registration functions like `registerAffinePasses`, the `PassWrapper`
machinery — is generated by *tablegen* from a five-line declaration in
[`Passes.td`](../lib/Transform/Affine/Passes.td). Migrating from the
hand-written form to this one is precisely the subject of Tutorial 4; today,
just note the mapping: `getArgument()` ↔ the name in `Passes.td`,
`PassWrapper<...>` ↔ the generated base class. One real difference: the
tablegen declaration doesn't specify an anchor, making this a *generic*
pass that runs once on the whole module — so `getOperation()` returns a
generic `Operation*`, and the code below writes `getOperation()->walk`
(arrow) where the article's `FuncOp` version wrote `getOperation().walk`
(dot).

> **Want to build and run this tutorial's exact code?** The hand-written
> `PassWrapper` version of all three passes — plus the article-era
> `tutorial-opt.cpp` with its explicit `PassRegistration` lines — is
> preserved, buildable, in [`tutorial/code/03/`](code/03/):
>
> ```bash
> bazel build //tutorial/code/03:tutorial-opt-03
> bazel-bin/tutorial/code/03/tutorial-opt-03 tests/affine_loop_unroll.mlir --affine-full-unroll
> ```
>
> It registers the same flags and passes the same FileCheck assertions as
> `tutorial-opt` on this tutorial's tests. (It has to be a separate binary:
> two passes cannot register the same `--affine-full-unroll` flag in one
> tool.) Every command in this tutorial works with either binary — see
> [`tutorial/code/README.md`](code/README.md) for the conventions.

## 5. Implementing the pass: walking the IR

Here is the entire implementation:

```cpp
using mlir::affine::AffineForOp;
using mlir::affine::loopUnrollFull;

void runOnOperation() {
  getOperation()->walk([&](AffineForOp op) {
    if (failed(loopUnrollFull(op))) {
      op.emitError("unrolling failed");
      signalPassFailure();
    }
  });
}
```

Reading it inside out:

- `walk(callback)` — available on every operation — traverses everything
  nested inside the operation, in post-order (children before parents; for
  nested loops, inner loop first, which is exactly what unrolling wants).
  The callback's argument type doubles as a *filter*: because the lambda
  takes an `AffineForOp`, it is invoked only for `affine.for` ops, skipping
  everything else. This "walk and match by type" idiom is the simplest way
  to visit IR.
- `loopUnrollFull(op)` is the upstream utility doing the real work. It
  returns a `LogicalResult`, MLIR's success/failure type — the same one
  `matchAndRewrite` returned in upstream patterns you've seen; produce one
  with `success()`/`failure()` and test one with `failed(...)`.
- On failure, `op.emitError(...)` attaches a diagnostic to the offending
  operation — MLIR prints it with source location, like a compiler error —
  and `signalPassFailure()` is the official way to tell the pass manager
  this pass failed, aborting the pipeline.

Note what we did *not* write: no iteration over functions, no manual
recursion into loop bodies, no worklist. `walk` handles traversal, and the
pass manager handles everything outside `runOnOperation`.

## 6. Step: run the unrolling pass

The test input
[`tests/affine_loop_unroll.mlir`](../tests/affine_loop_unroll.mlir) sums
the 4 entries of a buffer (the `iter_args` loop-carried-sum idiom is
explained in Tutorial 1 §4):

```mlir
func.func @test_single_nested_loop(%buffer: memref<4xi32>) -> (i32) {
  %sum_0 = arith.constant 0 : i32
  %sum = affine.for %i = 0 to 4 iter_args(%sum_iter = %sum_0) -> i32 {
    %t = affine.load %buffer[%i] : memref<4xi32>
    %sum_next = arith.addi %sum_iter, %t : i32
    affine.yield %sum_next : i32
  }
  return %sum : i32
}
```

(`memref` is a new dialect/type: a *memory reference*, MLIR's typed view of
a buffer in memory — here, 4 `i32`s — with `affine.load` reading an element
at an index. It's the bridge between SSA values and actual memory.)

Run the pass on it:

```bash
$TUTORIAL_OPT tests/affine_loop_unroll.mlir --affine-full-unroll
```

Output for that function:

```mlir
#map = affine_map<(d0) -> (d0 + 1)>
#map1 = affine_map<(d0) -> (d0 + 2)>
#map2 = affine_map<(d0) -> (d0 + 3)>
module {
  func.func @test_single_nested_loop(%arg0: memref<4xi32>) -> i32 {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = affine.load %arg0[%c0] : memref<4xi32>
    %1 = arith.addi %c0_i32, %0 : i32
    %2 = affine.apply #map(%c0)
    %3 = affine.load %arg0[%2] : memref<4xi32>
    %4 = arith.addi %1, %3 : i32
    %5 = affine.apply #map1(%c0)
    %6 = affine.load %arg0[%5] : memref<4xi32>
    %7 = arith.addi %4, %6 : i32
    %8 = affine.apply #map2(%c0)
    %9 = affine.load %arg0[%8] : memref<4xi32>
    %10 = arith.addi %7, %9 : i32
    return %10 : i32
  }
}
```

The loop is gone: four loads and four adds, one per iteration. The
`affine_map` lines at the top — those affine functions from section 1, now
in the flesh — and the `affine.apply` ops express the per-iteration index
arithmetic (`i+1`, `i+2`, `i+3`) symbolically. They're clutter here
(`%c0` is a constant, so each index is too), but the unroll utility doesn't
clean up after itself. Hold that thought for the next section.

## 7. The same pass with the pattern rewrite engine

Walking the IR by hand scales poorly: real rewrites must worry about
*order* (does unrolling one loop expose another?) and *iteration* (keep
going until nothing changes). MLIR's **pattern rewrite engine** takes over
that bookkeeping. You provide *patterns* — small classes that say "I match
this kind of op, and here is how I rewrite it" — and a *driver* applies
them until no pattern matches anymore.

[`AffineFullUnrollPatternRewrite.cpp`](../lib/Transform/Affine/AffineFullUnrollPatternRewrite.cpp)
re-implements unrolling this way:

```cpp
// A pattern that matches on AffineForOp and unrolls it.
struct AffineFullUnrollPattern : public OpRewritePattern<AffineForOp> {
  AffineFullUnrollPattern(mlir::MLIRContext *context)
      : OpRewritePattern<AffineForOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(AffineForOp op,
                                PatternRewriter &rewriter) const override {
    // This is technically not allowed, since in a RewritePattern all
    // modifications to the IR are supposed to go through the `rewriter` arg,
    // but it works for our limited test cases.
    return loopUnrollFull(op);
  }
};

// A pass that invokes the pattern rewrite engine.
struct AffineFullUnrollPatternRewrite
    : impl::AffineFullUnrollPatternRewriteBase<AffineFullUnrollPatternRewrite> {
  using AffineFullUnrollPatternRewriteBase::AffineFullUnrollPatternRewriteBase;
  void runOnOperation() {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<AffineFullUnrollPattern>(&getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};
```

Piece by piece:

- `OpRewritePattern<AffineForOp>` — a pattern that matches one op type.
  `matchAndRewrite` does both jobs: return `failure()` to say "I don't
  match" (leaving the IR untouched), or rewrite the IR and return
  `success()`.
- `benefit` — the pattern's priority. When several patterns match the same
  op, the driver tries higher-benefit ones first. With one pattern it's
  moot; it earns its keep in section 8.
- The pass body shrinks to: collect patterns in a `RewritePatternSet`, hand
  them to `applyPatternsAndFoldGreedily`. The *greedy driver* repeatedly
  scans for matching patterns and applies them until a **fixed point** — a
  full sweep in which nothing matches. As the "AndFold" says, it also runs
  op *folders* along the way: built-in per-op simplifications like
  `0 + x → x` or evaluating constant expressions.
- Note the source comment: an honest rewrite pattern performs all IR
  changes through the `rewriter` argument (as we do in section 8), so the
  driver can track what changed and re-examine affected ops. Calling
  `loopUnrollFull` behind the driver's back is a shortcut that happens to
  survive our tests.

Run it and compare with section 6:

```bash
$TUTORIAL_OPT tests/affine_loop_unroll.mlir --affine-full-unroll-rewrite
```

```mlir
module {
  func.func @test_single_nested_loop(%arg0: memref<4xi32>) -> i32 {
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = affine.load %arg0[%c0] : memref<4xi32>
    %1 = affine.load %arg0[%c1] : memref<4xi32>
    %2 = arith.addi %0, %1 : i32
    %3 = affine.load %arg0[%c2] : memref<4xi32>
    %4 = arith.addi %2, %3 : i32
    %5 = affine.load %arg0[%c3] : memref<4xi32>
    %6 = arith.addi %4, %5 : i32
    return %6 : i32
  }
}
```

Same unrolling — but cleaner. The driver's folding already collapsed the
`affine.apply` index arithmetic into constants and deleted the useless
`0 + %0` add from the first iteration. You get standard cleanups for free
just by going through the engine.

## 8. A real rewrite: multiplications into additions

The unroll pattern delegated to an upstream utility. Now we build patterns
that construct IR themselves, using the `rewriter` API properly — this is
where you learn the vocabulary used by every rewrite pattern ever written.

The `--mul-to-add` pass replaces multiplication-by-constant with repeated
addition (imagine a target where multiplication is much more expensive than
addition). Two cooperating patterns live in
[`lib/Transform/Arith/MulToAdd.cpp`](../lib/Transform/Arith/MulToAdd.cpp):

- **PowerOfTwoExpand** (benefit 2): `y = C*x` → `y = (C/2)*x + (C/2)*x`
  when `C` is a power of two.
- **PeelFromMul** (benefit 1): `y = C*x` → `y = (C-1)*x + x` otherwise.

Applied repeatedly, `9*x` becomes `8*x + x`, then the `8*x` halves its way
down to single additions — about `log C` additions total rather than the
`C` you'd get from peeling alone. Here is `PowerOfTwoExpand`:

```cpp
struct PowerOfTwoExpand : public OpRewritePattern<MulIOp> {
  PowerOfTwoExpand(mlir::MLIRContext *context)
      : OpRewritePattern<MulIOp>(context, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(MulIOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getOperand(0);

    // canonicalization patterns ensure the constant is on the right, if there
    // is a constant
    Value rhs = op.getOperand(1);
    auto rhsDefiningOp = rhs.getDefiningOp<arith::ConstantIntOp>();
    if (!rhsDefiningOp) {
      return failure();
    }

    int64_t value = rhsDefiningOp.value();
    bool is_power_of_two = (value & (value - 1)) == 0;

    if (!is_power_of_two) {
      return failure();
    }

    ConstantOp newConstant = rewriter.create<ConstantOp>(
        rhsDefiningOp.getLoc(),
        rewriter.getIntegerAttr(rhs.getType(), value / 2));
    MulIOp newMul = rewriter.create<MulIOp>(op.getLoc(), lhs, newConstant);
    AddIOp newAdd = rewriter.create<AddIOp>(op.getLoc(), newMul, newMul);

    rewriter.replaceOp(op, newAdd);
    rewriter.eraseOp(rhsDefiningOp);

    return success();
  }
};
```

Walk through it as two phases:

**Match phase** — may only *inspect*:

- `op.getOperand(n)` fetches an input `Value`.
- `value.getDefiningOp<OpTy>()` walks *backwards* through the SSA graph to
  the operation that produced the value — returning null if that operation
  isn't an `OpTy`. Here it asks: "is the right operand a constant
  integer?" (The source comment explains why checking only the *right*
  operand suffices: MLIR's global canonicalization rules move constants to
  the right of commutative ops, so `9*x` and `x*9` both arrive as
  `muli %x, %c9`.)
- `(value & (value - 1)) == 0` is the classic power-of-two bit trick.
- Every non-match exits with `failure()` **before any mutation**. This is
  the pattern-writing contract: a pattern must not touch the IR and then
  bail, or the driver's bookkeeping breaks.

**Rewrite phase** — every change goes through `rewriter`:

- `rewriter.create<OpTy>(loc, ...)` builds a new op. Every op carries a
  *location* (`getLoc()`) — MLIR's source-tracking breadcrumb, used to
  produce meaningful diagnostics after many rounds of rewriting; reusing
  the original ops' locations keeps that chain intact.
- `rewriter.getIntegerAttr(type, value)` builds the compile-time constant
  *attribute* the new `ConstantOp` holds (attributes are how ops store
  static data, as opposed to runtime SSA operands).
- `rewriter.replaceOp(op, newAdd)` rewires all uses of `op`'s results to
  `newAdd`'s results and deletes `op` — the moral equivalent of "return the
  new expression."
- `rewriter.eraseOp(rhsDefiningOp)` deletes the now-unused constant. (If it
  were still used elsewhere this would be wrong — a real pass would let
  dead-code elimination handle it; here it's safe for the test cases.)

`PeelFromMul` is nearly identical, with a lower benefit and no
power-of-two check:

```cpp
    ConstantOp newConstant = rewriter.create<ConstantOp>(
        rhsDefiningOp.getLoc(),
        rewriter.getIntegerAttr(rhs.getType(), value - 1));
    MulIOp newMul = rewriter.create<MulIOp>(op.getLoc(), lhs, newConstant);
    AddIOp newAdd = rewriter.create<AddIOp>(op.getLoc(), newMul, lhs);

    rewriter.replaceOp(op, newAdd);
    rewriter.eraseOp(rhsDefiningOp);
```

The **benefit values encode the strategy**: when `C` is a power of two,
*both* patterns match the op, and benefit makes the driver prefer halving
(log-many steps) over peeling (linearly many). It also lets `PeelFromMul`
skip re-checking: its body notes it is *guaranteed* `C` is not a power of
two, because the higher-benefit pattern already had its chance. The pass
itself is just the two-line registration you saw in section 7:

```cpp
struct MulToAdd : impl::MulToAddBase<MulToAdd> {
  using MulToAddBase::MulToAddBase;

  void runOnOperation() {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<PowerOfTwoExpand>(&getContext());
    patterns.add<PeelFromMul>(&getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};
```

Run it on [`tests/mul_to_add.mlir`](../tests/mul_to_add.mlir), which
multiplies by 8 and by 9:

```bash
$TUTORIAL_OPT tests/mul_to_add.mlir --mul-to-add
```

```mlir
module {
  func.func @just_power_of_two(%arg0: i32) -> i32 {
    %0 = arith.addi %arg0, %arg0 : i32
    %1 = arith.addi %0, %0 : i32
    %2 = arith.addi %1, %1 : i32
    return %2 : i32
  }
  func.func @power_of_two_plus_one(%arg0: i32) -> i32 {
    %0 = arith.addi %arg0, %arg0 : i32
    %1 = arith.addi %0, %0 : i32
    %2 = arith.addi %1, %1 : i32
    %3 = arith.addi %2, %arg0 : i32
    return %3 : i32
  }
}
```

Trace the second function yourself: `9*x` → peel → `8*x + x` → three
halvings → the doubling chain `%0, %1, %2`, plus the final `+ %arg0` from
the peel. All multiplications are gone, reached purely by local patterns
running to a fixed point.

## 9. Walk or pattern rewrite?

- **Pattern rewrite** when the transformation is *local*: you can decide to
  fire by looking at one op and its neighborhood (operands' defining ops,
  users), and applying it repeatedly converges. You get
  iteration-to-fixed-point, pattern prioritization, and folding for free.
- **Walk** when you need *global* context — analyses that span whole
  functions (think common-subexpression elimination, which must reason
  about the dataflow of the entire program), custom traversal orders, or
  one-shot structural surgery. You take on the iteration logic yourself.

Most passes in the wild are pattern-based; the walk is the escape hatch.

## 10. How the pass gets into the binary

The chain from `.cpp` to `--flag` is worth seeing once. With Bazel, each
pass is a `cc_library` in
[`lib/Transform/Affine/BUILD`](../lib/Transform/Affine/BUILD) (trimmed):

```python
cc_library(
    name = "AffineFullUnroll",
    srcs = ["AffineFullUnroll.cpp"],
    hdrs = ["AffineFullUnroll.h", "Passes.h"],
    deps = [
        ":pass_inc_gen",                       # tablegen output (Tutorial 4)
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineUtils",     # provides loopUnrollFull
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)
```

and the binary in [`tools/BUILD`](../tools/BUILD) depends on the pass
libraries plus MLIR's opt-tool library:

```python
cc_binary(
    name = "tutorial-opt",
    srcs = ["tutorial-opt.cpp"],
    deps = [
        "//lib/Transform/Affine:Passes",
        "//lib/Transform/Arith:Passes",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Pass",
        # ... deps for later tutorials trimmed ...
    ],
)
```

The CMake equivalents are
[`lib/Transform/Affine/CMakeLists.txt`](../lib/Transform/Affine/CMakeLists.txt)
(`add_mlir_library(AffineFullUnroll ...)`) and
[`tools/CMakeLists.txt`](../tools/CMakeLists.txt)
(`add_llvm_executable(tutorial-opt ...)` linking the pass libraries). When
you add a new pass, you touch: the pass files, its `BUILD`/`CMakeLists.txt`
library entry, and a registration call in `tutorial-opt.cpp` — that's the
whole checklist.

## 11. Step: test the passes

The lit/FileCheck machinery from Tutorial 2 applies unchanged — the only
news is that `RUN` lines now invoke `tutorial-opt`, which is on lit's
`$PATH` because the Bazel `test_utilities` filegroup includes
`//tools:tutorial-opt` (CMake: the lit config puts the build's `tools/` dir
on the `$PATH`). From
[`tests/affine_loop_unroll.mlir`](../tests/affine_loop_unroll.mlir):

```mlir
// RUN: tutorial-opt %s --affine-full-unroll > %t
// RUN: FileCheck %s < %t

// RUN: tutorial-opt %s --affine-full-unroll-rewrite > %t
// RUN: FileCheck %s < %t

func.func @test_single_nested_loop(%buffer: memref<4xi32>) -> (i32) {
  %sum_0 = arith.constant 0 : i32
  // CHECK-LABEL: test_single_nested_loop
  // CHECK-NOT: affine.for
  ...
```

One file tests both pass variants: two RUN pipelines, same CHECK
assertions — "whatever else you produce, no `affine.for` may survive." A
deliberately loose test (Tutorial 2's trade-off discussion): it stays green
across both implementations and any future cleanup changes. The
`mul_to_add.mlir` test is the opposite call — it pins the exact addition
chain with the capture-variable style from Tutorial 2 §5, because the
addition structure *is* the pass's contract. The file also has a second
test function with a *nested* loop, checking that `walk` reaches inner
loops.

Run them:

```bash
# Bazel
bazel test //tests:affine_loop_unroll.mlir.test //tests:mul_to_add.mlir.test

# CMake
llvm-lit -sv build-ninja/tests --filter 'affine_loop_unroll|mul_to_add'
```

```
Total Discovered Tests: 18
  Excluded: 15 (83.33%)
  Passed  :  3 (16.67%)
```

## 12. Bonus: developer tooling

Two quality-of-life notes from the article, both still wired up in this
repo:

- **IDE support (clangd).** C++ editing without a language server is
  painful, and clangd needs a `compile_commands.json`. For Bazel, the repo
  includes [Hedron's compile-commands
  extractor](https://github.com/hedronvision/bazel-compile-commands-extractor)
  (see `MODULE.bazel`): run `bazel run @hedron_compile_commands//:refresh_all`
  to (re)generate it, and rerun after editing any `BUILD` file. For CMake,
  `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` does the same job.
- **CI.** `.github/workflows/build_and_test.yml` (and a CMake twin) builds
  and tests every push. The Bazel cache action matters: a cold LLVM build
  takes 1–2 hours in CI, a cached one a few minutes — the same economics
  you experience locally.

## Differences from the original article

- **The pass boilerplate is generated now.** The article hand-writes each
  pass as the `PassWrapper` subclass shown in section 4; the repo's current
  code declares passes in `Passes.td` and inherits from
  tablegen-generated `impl::<Name>Base` classes, and registration happens
  via generated `register*Passes()` functions instead of explicit
  `PassRegistration<...>()` lines. The `runOnOperation` bodies are
  unchanged. The migration is exactly the subject of the next
  article/tutorial. The article-style, tablegen-free version of this
  tutorial's code is kept buildable in [`tutorial/code/03/`](code/03/)
  (see the section 4 box).
- **The passes are no longer anchored to `func.func`.** The article defines
  `OperationPass<FuncOp>`; the current tablegen declarations are generic
  passes, which is why the code reads `getOperation()->walk` (an
  `Operation*`) rather than the article's `getOperation().walk` (a
  `FuncOp`). The anchoring *concept* (section 4) is unchanged and returns
  in later tutorials.
- `lib/Transform/Arith` additionally contains a PDLL variant of MulToAdd
  (`--mul-to-add-pdll`) from a much later article — ignore it for now.

## Where to go next

You have now written (well, read — the exercises fix that) both styles of
pass, know why passes are anchored, and have the rewriter vocabulary: match
via `getDefiningOp`, rewrite via `rewriter.create`/`replaceOp`/`eraseOp`,
and let the greedy driver handle iteration. Next,
[Tutorial 4: Using Tablegen for Passes](04-using-tablegen-for-passes.md)
covers the tablegen machinery that generated those `impl::...Base`
classes — `Passes.td`, `Passes.h.inc`, and the `pass_inc_gen` build rule
from section 10 — eliminating the last of the boilerplate.

**Exercises**

1. Predict the output of `--mul-to-add` for multiplication by 7 (write the
   pattern applications out by hand: which pattern fires at each step, and
   why?). Then check yourself: change the constant in a copy of
   `tests/mul_to_add.mlir` and run `$TUTORIAL_OPT` on it.
2. What about `C = 1` or `C = 0`? Trace the code:
   `(value & (value - 1)) == 0` is true for both, so *PowerOfTwoExpand*
   claims them and would rewrite `1*x` into `0*x + 0*x` — heading somewhere
   unpleasant. Now run it. The actual output is just `return %arg0`
   (resp. a constant 0): the greedy driver's built-in **folders** simplify
   `1*x` and `0*x` away before the pattern ever fires. Safety nets like
   this are part of why rewrites go through the driver.
3. Rerun `--mul-to-add` with `--mlir-print-ir-after-all` to see the IR
   after each pass; then try `--debug` if your build has assertions enabled
   to see each pattern application the greedy driver attempts.
4. Modify the walk-based pass to unroll only loops whose trip count exceeds
   a threshold (peek at `AffineForOp`'s methods like
   `getConstantUpperBound` in the upstream headers), rebuild, and watch the
   test fail — then adjust the test.
5. The doubly nested loop in `affine_loop_unroll.mlir` unrolls to 12 loads.
   Why does post-order traversal matter for it? What would happen on a
   pre-order walk that unrolled the outer loop first? (The answer is "it
   still works, but think about why" — the unrolled copies of the inner
   loop are new ops the walk has to find.)
