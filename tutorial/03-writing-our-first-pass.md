# Chapter 3: Writing Our First Pass

This chapter picks up where [Chapter 2](02-testing-a-lowering.md) left off:
we can run upstream passes and test them — now we write passes of our own,
organize them into a project, and build `tutorial-opt`, this project's own
version of `mlir-opt` that carries them.

> **Heads-up before you read any code:** the pass code this chapter
> teaches is the original hand-written form, and it is **not what
> `lib/` contains** — the main tree has since migrated the same passes
> to tablegen-generated boilerplate (that migration is
> [Chapter 4](04-using-tablegen-for-passes.md)'s subject). The
> hand-written form is preserved, buildable, in
> [`tutorial/code/03/`](code/03/) (`bazel build
> //tutorial/code/03:tutorial-opt-03`); it registers the same flags and
> produces the same output as `tutorial-opt` on this chapter's tests, so
> every command below works with either binary. A short
> [What the repo has now](#what-the-repo-has-now) note at the end of
> section 5 summarizes how `lib/`'s versions differ.

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

**Prerequisites:** Chapters [1](01-mlir-basics-and-running-a-lowering.md)
and [2](02-testing-a-lowering.md), and a build that can produce
`tutorial-opt`. With Bazel that is automatic (the commands below build it on
first use). With CMake, `tutorial-opt` only builds against the LLVM
**submodule** — you must have checked out and built
`externals/llvm-project` per the README; a system-installed LLVM will not
work (see Chapter 2's CMake notes).

Set a shell variable once so the commands below work for either build
system, mirroring Chapter 2's pattern:

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
Chapters 1–2 were passes; so are classic optimizations. A pass doesn't
have to lower anything: it can rewrite IR *within* one abstraction level,
which is what both passes in this chapter do.

Our first example operates on the
**[`affine` dialect](https://mlir.llvm.org/docs/Dialects/Affine/)** — the
highest-level loop dialect in MLIR, and the reason Chapter 1 said "keep the
program at the right abstraction level for each optimization." Affine loops
are loops whose bounds and index expressions are *affine functions* (sums of
variables times constants, like `2*i + j + 5`) of loop variables and
symbols. That restriction is what the dialect sells: it was originally
built for polyhedral loop analysis — a mathematical framework that studies
loop nests via lattice theory — and it makes questions like "what is this
loop's trip count?" statically answerable. An `scf.for` loop can have any
dynamic bound; an
[`affine.for`](https://mlir.llvm.org/docs/Dialects/Affine/#affinefor-affineaffineforop)
loop is analyzable by construction.

We exploit exactly that: our pass **fully unrolls** every affine loop —
replacing the loop with one copy of its body per iteration. Unrolling is a
real optimization (it removes branch overhead and exposes parallelism, and
in some domains — like the fully homomorphic encryption (FHE) compilers 
this book builds toward — loops must be unrolled entirely because the 
target can't branch at all). But honestly, we picked it because MLIR 
provides a one-call utility, `loopUnrollFull`, that does the work. 
Both passes in this chapter are deliberately thin: the point is to learn 
how to *set up* a pass, navigate the IR through the C++ API, and modify it 
— not to write a clever algorithm.

## 2. Project organization

Before writing code, decide where it goes. A typical MLIR codebase (the
LLVM monorepo included) splits code into two parallel directory trees:

- `include/` — header files and tablegen files, defining the public
  interface;
- `lib/` — the implementation code.

Within each, conventional subdirectories signal a component's role:

- `Transform/` — passes that transform code *within* a dialect (this
  chapter);
- `Conversion/` — passes that convert *between* dialects (the lowerings of
  Chapter 1; this repo's `lib/Conversion/PolyToStandard` arrives in a
  later chapter);
- `Analysis/` — passes that compute information without modifying IR;
- `Dialect/` — definitions of the dialects themselves.

This repo deviates in one deliberate way: **`include/` and `lib/` are
merged** — every header sits next to its implementation file. The parallel
tree exists in ordinary C++ projects partly to separate public interface
from private implementation, but Bazel has finer-grained ways to enforce
visibility, navigating mirrored trees is tedious, and this is a teaching
codebase — simpler is better. So the layout relevant to this chapter is:

```
lib/Transform/Affine/
├── AffineFullUnroll.h                    # walk-based pass (this chapter)
├── AffineFullUnroll.cpp
├── AffineFullUnrollPatternRewrite.h      # pattern-based pass (this chapter)
├── AffineFullUnrollPatternRewrite.cpp
├── Passes.td, Passes.h                   # tablegen boilerplate (Chapter 4)
├── BUILD                                 # Bazel targets
└── CMakeLists.txt
lib/Transform/Arith/
├── MulToAdd.h, MulToAdd.cpp              # this chapter
├── ...                                   # PDLL variant, from a later chapter
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
the parts that matter for this chapter (the full file also registers
dialects and a pipeline from later chapters):

***tools/tutorial-opt.cpp***
```cpp
#include "lib/Transform/Affine/Passes.h"
#include "lib/Transform/Arith/Passes.h"
#include "mlir/include/mlir/InitAllDialects.h"
#include "mlir/include/mlir/InitAllPasses.h"
#include "mlir/include/mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();

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
- `registerAllPasses()` does the same for every stock MLIR pass — the
  reason for the hundreds of extra `--help` flags you'll see below.
- `registerAffinePasses()` / `registerArithPasses()` are generated
  functions (one per pass group; more below) that make our passes visible
  to the pass registry — this is what turns them into `--affine-full-unroll`
  style flags.
- `MlirOptMain(...)` runs the whole show; `asMainReturnCode` converts its
  `LogicalResult` into a process exit code.

At this chapter's stage, with no generated registration functions yet, the
same hookup is an explicit `PassRegistration` line per pass. That version
of the driver is preserved — and buildable — as
[`tutorial/code/03/tutorial-opt.cpp`](code/03/tutorial-opt.cpp); here it is
in full (minus the orientation comment at the top of the file):

***tutorial/code/03/tutorial-opt.cpp***
```cpp
#include "tutorial/code/03/AffineFullUnroll.h"
#include "tutorial/code/03/AffineFullUnrollPatternRewrite.h"
#include "tutorial/code/03/MulToAdd.h"
#include "mlir/include/mlir/InitAllDialects.h"
#include "mlir/include/mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);

  mlir::PassRegistration<mlir::tutorial::AffineFullUnrollPass>();
  mlir::PassRegistration<mlir::tutorial::AffineFullUnrollPassAsPatternRewrite>();
  mlir::PassRegistration<mlir::tutorial::MulToAddPass>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Tutorial Pass Driver", registry));
}
```

(The original code included the pass headers from `lib/Transform/...`;
the preserved copy includes them from its own directory, because `lib/`'s
headers have since been migrated — that migration is Chapter 4, and the
classes this driver registers, like `AffineFullUnrollPass`, are section
4's subject. The heads-up at the top of this chapter has the build
command for this version. One more visible difference: this driver has no
`registerAllPasses()`, so `tutorial-opt-03 --help` lists *only* our three
passes — the upstream-pass buffet is a later addition to the repo's
driver.)

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

Three of these are this chapter's subject (`--mul-to-add-pdll` belongs to
a much later chapter). `--help` also lists hundreds of upstream passes:
your tool is a superset of `mlir-opt`.

## 4. Anatomy of a pass

This chapter defines the pass the from-scratch way, and it's worth
understanding in full even though the repo has since moved to generated
boilerplate (we'll get to that). In general, a pass is a C++ class that
supplies three functions — one that does the work, two that name and
describe its CLI flag — and inherits everything else the pass API
requires from a base class:


***tutorial/code/03/AffineFullUnroll.h***
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
The three functions we need to implement are:
- `runOnOperation()` is the heart of every pass — the method the pass
  manager calls to do the work.
- `getArgument()` is the CLI flag name.
- `getDescription()` is the CLI description when running `--help` on the `mlir-opt` like tool.

Everything else is inherited. `PassWrapper` fills in the rest of the
required pass API (such as a compliant clone/copy method — why a pass
must be *copyable* becomes clear just below) using the *Curiously
Recurring Template Pattern* — that's why the class passes itself as the
first template argument. CRTP lets the base class call methods of the
derived class without virtual dispatch; if it looks odd, treat it as an
idiom to copy, not master.

The second template argument deserves its own paragraph.
`OperationPass<mlir::func::FuncOp>` **anchors** the pass to function
operations: the pass manager will invoke `runOnOperation()` once per
`func.func` in the module, and inside the pass, `getOperation()` returns
the `FuncOp` currently being processed. (The MLIR docs formally describe
[what is required of an
OperationPass](https://mlir.llvm.org/docs/PassManagement/#operation-pass-static-filtering-by-op-type),
and in particular limit this anchoring to specific operations like
functions and modules.)



### Why anchor at all? Parallelism

Be careful what "parallelism" means here: it is *not* that different
passes run at the same time — a pipeline is strictly sequential
(canonicalize finishes, *then* cse starts). What runs concurrently is
**one pass, applied to many separate ops of its anchor type**. Anchoring on `func.func` turns "run this pass" into one
independent job per function in the module, and the pass manager hands
those jobs to a thread pool (MLIR's context owns one; threading is on by
default):

```
module { func.func @f   func.func @g   func.func @h }

thread 1:  runOnOperation() on @f
thread 2:  runOnOperation() on @g    <- same pass, three simultaneous calls
thread 3:  runOnOperation() on @h
────────────────── join ──────────────────
(only then does the next pass in the pipeline start)
```

Each thread does ordinary single-threaded work on its own private subtree
of the IR — the model is `make -j` compiling independent `.c` files, with
the pass manager as the scheduler. Two things you've already seen exist
*because* of this fan-out: the pass manager doesn't share one pass object
across threads, it **clones** the pass once per thread — that is what
`PassWrapper`'s copy machinery above is for — and every `MlirOptMain`
tool accepts `--mlir-disable-threading` to force one function at a time
(handy when three threads' debug output interleaves).

And like `make -j`, no locks are taken anywhere: the *independence of the
jobs* is the entire safety argument. Two conditions make the jobs truly
independent, and the pass infrastructure enforces both:

1. **Your pass code must stay inside the op it was given.** Thread 1 is
   processing `@f`; if the pass reached over to inspect or edit `@g`
   ("let me check how `@g` calls this…"), it would be touching IR that
   thread 2 is mutating at that very moment — a data race. A pass that
   genuinely needs a whole-program view (inlining, say, which edits
   callers and callees together) anchors on the module instead: there is
   only one module, so one job on one thread. Correct, but serial —
   that is the price.

2. **Nothing inside the anchor op may be wired to anything outside it.**
   This one is subtle, because in MLIR "editing inside X" can physically
   write to memory *outside* X. SSA edges are two-way links: every
   `Value` keeps a **use-list**, and creating or erasing an op that uses
   `%x` writes to `%x`'s use-list. So if an op inside `@f`'s body could
   use a value defined outside `@f`, then a pass editing only `@f`'s
   interior would still be mutating shared state — even though its code
   never "looked" outside. The trait that rules this out is
   `IsolatedFromAbove`: nothing inside a `func.func` may reference an
   SSA value defined above it, so a function's interior is genuinely
   private to whichever thread holds it. The pass manager refuses to
   run a parallel pass anchored on any op without this trait.

The second condition explains why we can't just anchor the pass on
`affine.for` and skip the walk below. A loop is wired to its
surroundings in every direction (a sketch; section 6's real test input
has all three edges):

```mlir
%c = arith.constant 0 : i32               // defined outside the loop...
%r = affine.for %i = 0 to 4 iter_args(%acc = %c) -> i32 {
  %t = arith.addi %acc, %c : i32          // ...used inside the body
  affine.yield %t : i32
}
return %r : i32                           // loop result used outside
```

— and unrolling doesn't even stay inside the loop: it *deletes the
`affine.for` op itself* and splices the copied bodies into the parent
block, rewriting the uses of `%r`. Two threads doing that to two loops in
the same function would be editing the same block and the same use-lists
concurrently. So `affine.for` is not `IsolatedFromAbove`, and MLIR won't
let you build a parallel pass on it. Hence the shape of passes like ours:
anchor at the isolation boundary (`func.func`), then *walk* — inside your
own function, where you are the only thread — down to the non-isolated
ops you actually care about.

One footnote so the trait doesn't overpromise: "a function can't touch
outside variables" sounds wrong if you're thinking of C, where functions
mutate globals all the time. `IsolatedFromAbove` seals only the **SSA
graph** (the `%`-values). Memory is a separate channel — two functions
can happily store to the same `memref` — which is part of why condition 1
remains a rule your code must follow, rather than something the IR's
structure could enforce for you.

## 5. Implementing the pass: walking the IR

The implementation has exactly two jobs: visit every `affine.for` nested
anywhere in the function, and invoke MLIR's unroll utility on each one it
finds. Both come as ready-made API — a traversal method and a one-call
utility — which is why the entire body fits in seven lines:

***tutorial/code/03/AffineFullUnroll.cpp***
```cpp
using mlir::affine::AffineForOp;
using mlir::affine::loopUnrollFull;

// A pass that manually walks the IR
void AffineFullUnrollPass::runOnOperation() {
  getOperation().walk([&](AffineForOp op) {
    if (failed(loopUnrollFull(op))) {
      op.emitError("unrolling failed");
      signalPassFailure();
    }
  });
}
```

`getOperation()` returns the `FuncOp` the pass is anchored on (section 4),
though we don't use any specific information about it being a function. We
instead call its `walk` method — present on all `Operation` instances —
which traverses the abstract syntax tree (AST) of the operation (here, the
function body) in *post-order*: children before parents, so for nested
loops the inner loop is visited first, which is exactly the order
unrolling wants. For each operation `walk` encounters, if that operation's
type matches the input type of the callback, the callback is executed —
because our lambda takes an `AffineForOp`, it runs only for `affine.for`
ops, and everything else is skipped. This "walk and match by type" idiom
is the simplest way to visit IR.

Inside the callback, we attempt to unroll the loop, and if that fails we
quit with a diagnostic error. `loopUnrollFull(op)` is the upstream utility
doing the real work; it returns a `LogicalResult`, MLIR's success/failure
type — the same one `matchAndRewrite` returned in the upstream patterns
you've seen (produce one with `success()`/`failure()`, test one with
`failed(...)`). On failure, `op.emitError(...)` attaches a diagnostic to
the offending operation — MLIR prints it with its source location, like a
compiler error — and `signalPassFailure()` is the official way to tell
the pass manager this pass failed, aborting the pipeline.

Note what we did *not* write: no iteration over functions, no manual
recursion into loop bodies, no worklist. `walk` handles traversal, and the
pass manager handles everything outside `runOnOperation`.

### What the repo has now

The main tree,
[`lib/Transform/Affine/AffineFullUnroll.cpp`](../lib/Transform/Affine/AffineFullUnroll.cpp),
has this same body inside a *tablegen-generated* shell —
`struct AffineFullUnroll : impl::AffineFullUnrollBase<AffineFullUnroll>` —
where the generated base class supplies the flag name, description, and
registration (the `registerAffinePasses()` from section 3), declared in
[`Passes.td`](../lib/Transform/Affine/Passes.td) instead of C++. One
visible difference in the body: the generated pass is *generic* (not
anchored on `func.func`), so `getOperation()` returns an `Operation*` and
the walk is spelled `getOperation()->walk` (arrow, not dot). The same goes
for the other two passes in this chapter; the anchoring *concept* from
section 4 is unchanged and returns in later chapters. How and why to
migrate is [Chapter 4](04-using-tablegen-for-passes.md)'s whole subject —
nothing more about it is needed today.

## 6. Step: run the unrolling pass

The test input
[`tests/affine_loop_unroll.mlir`](../tests/affine_loop_unroll.mlir) sums
the 4 entries of a buffer (the `iter_args` loop-carried-sum idiom is
explained in Chapter 1 §4; it exists to keep loops in compliance with SSA
form — for more on SSA in MLIR, see
[this MLIR doc](https://mlir.llvm.org/docs/LangRef/#high-level-structure)):

***tests/affine_loop_unroll.mlir***
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

Concretely, a rewrite pattern is a subclass of `OpRewritePattern` with a
single method to override, `matchAndRewrite`, which performs the
transformation. Its return value is the `LogicalResult` from section 5 —
a wrapper around a boolean, with `success()`/`failure()` to construct one
and `failed(...)` to test one. A relative worth meeting here is
`FailureOr<T>`: a subclass of `std::optional<T>` that interoperates with
`LogicalResult` through the presence or absence of a value — "either
failure, or the thing you computed."

Two contract points before reading the code. First, inside
`matchAndRewrite`, every mutation of the IR is supposed to go through the
[`PatternRewriter`](https://mlir.llvm.org/docs/PatternRewriter/#pattern-rewriter)
argument: the rewriter is what makes a pattern
*atomic*, guaranteeing the changes take effect only if the method runs to
the end and succeeds. (Our pattern below violates this —
`loopUnrollFull` has no variant that accepts a `PatternRewriter` — and
gets away with it for our limited test cases.) Second, the *pass* is
still anchored on `func.func`, but a pattern can match *any* op type: the
rewrite engine performs the walk we wrote by hand in section 5 (an
optional configuration struct can choose the walk order), collecting any
number of patterns from a `RewritePatternSet` and greedily applying
whichever ones match — in
[an order related to their `benefit`](https://mlir.llvm.org/docs/PatternRewriter/#greedy-pattern-rewrite-driver)
— until no operation matches, every applicable pattern returns failure, or
a large iteration limit trips to avoid infinite loops.

The preserved pass re-implements unrolling this way (the main tree's
[twin](../lib/Transform/Affine/AffineFullUnrollPatternRewrite.cpp) has the
identical pattern inside the generated shell — see section 5's note):

***tutorial/code/03/AffineFullUnrollPatternRewrite.cpp***
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
void AffineFullUnrollPassAsPatternRewrite::runOnOperation() {
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<AffineFullUnrollPattern>(&getContext());
  // One could use GreedyRewriteConfig here to slightly tweak the behavior of
  // the pattern application.
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}
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
[`tutorial/code/03/MulToAdd.cpp`](code/03/MulToAdd.cpp) (main-tree twin:
[`lib/Transform/Arith/MulToAdd.cpp`](../lib/Transform/Arith/MulToAdd.cpp) —
same two patterns, generated shell):

- **PowerOfTwoExpand** (benefit 2): `y = C*x` → `y = (C/2)*x + (C/2)*x`
  when `C` is a power of two.
- **PeelFromMul** (benefit 1): `y = C*x` → `y = (C-1)*x + x` otherwise.

Applied repeatedly, `9*x` becomes `8*x + x`, then the `8*x` halves its way
down to single additions — about `log C` additions total rather than the
`C` you'd get from peeling alone.

One general principle organizes every pattern body, and it's worth having
in mind before reading the code: a pattern runs in two phases. The
**match phase** may only *inspect* the IR — walk operands, check
properties — and must bail out with `failure()` before anything has been
touched; the **rewrite phase**, entered only once the match is certain,
makes every change through the `rewriter` argument (section 7's
atomicity contract). Here is `PowerOfTwoExpand`, read with that split in
mind:

***tutorial/code/03/MulToAdd.cpp***
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

The specifics, phase by phase:

**Match phase** — may only *inspect*:

- `op.getOperand(n)` fetches an input `Value` that is the type representing an SSA value (i.e., and MLIR variable)
- `rsh.getDefiningOp<OpTy()` walks *backwards* through the SSA graph to
  the operation that produced the value — returning null if the type cannot be converted. 
  Here it asks: "is the right operand a constant
  integer?" (The source comment explains why checking only the *right*
  operand suffices: MLIR's
  [global canonicalization rules](https://mlir.llvm.org/docs/Canonicalization/#globally-applied-rules)
  move constants to the right of commutative ops, so `9*x` and `x*9` both
  arrive as `muli %x, %c9`.)
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
  [*attribute*](https://mlir.llvm.org/docs/LangRef/#attributes) the new
  `ConstantOp` holds (attributes are how ops store static data, as opposed
  to runtime SSA operands — you can put strings or dictionaries in an
  attribute, but for `ConstantOp` it's just an int).
- `rewriter.replaceOp(op, newAdd)` rewires all uses of `op`'s results to
  `newAdd`'s results and deletes `op` — the moral equivalent of "return the
  new expression."
- `rewriter.eraseOp(rhsDefiningOp)` deletes the now-unused constant. (If it
  were still used elsewhere this would be wrong — a real pass would let
  dead-code elimination handle it; here it's safe for the test cases.)

`PeelFromMul` is nearly identical, with a lower benefit and no
power-of-two check:

***tutorial/code/03/MulToAdd.cpp*** (excerpt)
```cpp
    ConstantOp newConstant = rewriter.create<ConstantOp>(
        rhsDefiningOp.getLoc(),
        rewriter.getIntegerAttr(rhs.getType(), value - 1));  // value - 1 instead of value / 2
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
itself has the same shape you saw in section 7 — collect the patterns,
hand them to the greedy driver:

***tutorial/code/03/MulToAdd.cpp***
```cpp
void MulToAddPass::runOnOperation() {
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<PowerOfTwoExpand>(&getContext());
  patterns.add<PeelFromMul>(&getContext());
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}
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

With two ways to define a pass — walk the entire IR from the root
operation, or match and rewrite patterns through the rewrite engine — the
natural question is when to use which.

Notice first that the choice is not about *power*. The pattern engine
expresses a convenient subset of what a pass can do — conceptually
trivially so: anyone who can walk the whole tree can, with enough effort,
do anything at all, up to and including reimplementing the pattern
rewrite engine. The engine's case rests on convenience, and on history:
MLIR's own
[Generic DAG Rewriter Infrastructure Rationale](https://mlir.llvm.org/docs/Rationale/RationaleGenericDAGRewriter/)
lays out the motivation, distilled from a long line of pattern-matching
systems in the LLVM project and elsewhere.

What the engine is convenient *for* is **local** transformations.
"Local" means the situation you're rewriting can be recognized from a
small subset of the IR viewed as a directed acyclic graph — pragmatically,
anything you can detect by looking around at an op's neighbors in the
same block and applying some filtering logic. "Is this `exp` followed by
a `log`, with no other uses of the `exp`'s result?" is local.
`PowerOfTwoExpand`'s "is my right operand a constant, and is it a power
of two?" is local — one hop up the SSA graph and a bit test.

Some analyses and optimizations, by contrast, must construct the entire
dataflow of a program before they can act. Common subexpression
elimination is the classic example: deciding whether it's *cost-effective*
to pull a subexpression used in multiple places into one variable depends
on the operation's cost and on what keeping an extra value alive does to
memory access and register availability at that point in the program —
information no amount of local pattern-matching can see. For work like
that — or for custom traversal orders, or one-shot structural surgery —
you walk, and you own the iteration logic yourself.

The accumulated wisdom: when the transformation fits, the pattern engine
is usually the *easier* path. You don't write sprawling case/switch logic
over everything that might appear in the IR; you don't hand-roll the
"keep going until nothing changes" loop (the engine re-applies patterns
for you, with section 7's freebies — fixed-point iteration, benefit
prioritization, folding); and each pattern can be written in isolation,
trusting the engine to combine them appropriately — exactly how
`PowerOfTwoExpand` and `PeelFromMul` teamed up in section 8. Most
rewrites in the wild are pattern-based; the walk is the escape hatch.

## 10. How the pass gets into the binary

The chain from `.cpp` to `--flag` is worth seeing once. With Bazel, each
pass is a `cc_library` in
[`lib/Transform/Affine/BUILD`](../lib/Transform/Affine/BUILD):

***lib/Transform/Affine/BUILD*** (excerpt)
```python
cc_library(
    name = "AffineFullUnroll",
    srcs = ["AffineFullUnroll.cpp"],
    hdrs = ["AffineFullUnroll.h", "Passes.h"],
    deps = [
        ":pass_inc_gen",                       # tablegen output (Chapter 4)
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineUtils",     # provides loopUnrollFull
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)
```

and the binary in [`tools/BUILD`](../tools/BUILD) depends on the pass
libraries plus MLIR's opt-tool library:

***tools/BUILD*** (excerpt)
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
        # ... deps for later chapters trimmed ...
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
whole checklist. ([`tutorial/code/03/BUILD`](code/03/BUILD) is the same
wiring for the preserved version, minus the tablegen rule.)

## 11. Step: test the passes

The lit/FileCheck machinery from Chapter 2 applies unchanged — the only
news is that `RUN` lines now invoke `tutorial-opt`, which is on lit's
`$PATH` because the Bazel `test_utilities` filegroup includes
`//tools:tutorial-opt` (CMake: the lit config puts the build's `tools/` dir
on the `$PATH`). The test file's RUN header:

***tests/affine_loop_unroll.mlir*** (excerpt)
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
deliberately loose test (Chapter 2's trade-off discussion): it stays green
across both implementations and any future cleanup changes. The
`mul_to_add.mlir` test is the opposite call — it pins the exact addition
chain with the capture-variable style from Chapter 2 §5, because the
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

Two quality-of-life notes, both wired up in this repo:

- **IDE support (clangd).** C++ editing without a language server is
  painful, and clangd needs a `compile_commands.json`. For Bazel, the repo
  includes [Hedron's compile-commands
  extractor](https://github.com/hedronvision/bazel-compile-commands-extractor)
  (see `MODULE.bazel`): run `bazel run @hedron_compile_commands//:refresh_all`
  to (re)generate it, and rerun after editing any `BUILD` file — clangd and
  clang-tidy then find [the generated json
  file](https://clang.llvm.org/docs/JSONCompilationDatabase.html)
  automatically. For CMake, `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` does the
  same job.
- **CI.** `.github/workflows/build_and_test.yml` (and a CMake twin) builds
  and tests every push. The Bazel cache action matters: a cold LLVM build
  takes 1–2 hours in CI, a cached one a few minutes — the same economics
  you experience locally.

## Where to go next

You have now written (well, read — the exercises fix that) both styles of
pass, know why passes are anchored, and have the rewriter vocabulary: match
via `getDefiningOp`, rewrite via `rewriter.create`/`replaceOp`/`eraseOp`,
and let the greedy driver handle iteration. Next,
[Chapter 4: Using Tablegen for Passes](04-using-tablegen-for-passes.md)
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
