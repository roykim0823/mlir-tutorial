# Tutorial 4: Using Tablegen for Passes

This is a step-by-step companion to the article
[Using Tablegen for Passes](https://jeremykun.com/2023/08/10/mlir-using-tablegen-for-passes/).
[Tutorial 3](03-writing-our-first-pass.md) ended with an IOU: the passes we
read inherited from mysterious `impl::AffineFullUnrollBase` classes, and
registration happened through functions like `registerAffinePasses()` that
we never wrote. This tutorial pays the debt — those are all *generated* by
a tool called **tablegen**, and here we read the 22 lines that produce them,
run the generator by hand, and read every line it emits.

Unlike earlier tutorials, this one requires almost no un-learning between
the article and the repo: the repo *is* the result of the article's
migration, so the files we read are the article's end state.

**What you will learn:**

- What tablegen is, and the right mental model for it ("see-through code
  generator", not abstraction layer).
- The anatomy of `Passes.td` — `def`, `Pass<"...">`, `summary`,
  `description`, `dependentDialects`.
- How to run `mlir-tblgen` yourself, and how Bazel/CMake wire it into the
  build.
- How to read the generated `Passes.h.inc`: the `GEN_PASS_DECL`,
  `GEN_PASS_DEF`, and `GEN_PASS_REGISTRATION` sections, and the
  `#define`/`#include` idiom that consumes them.
- What the migration deleted from Tutorial 3's hand-written pass, and what
  tablegen offers beyond it (doc generation, pass options).
- What tablegen's failure modes look like — with real error messages.

**Prerequisites:** [Tutorial 3](03-writing-our-first-pass.md). No new build
requirements: everything here can be verified with `mlir-tblgen` alone,
which ships with any MLIR build (`bazel-bin` after building any tablegen
target, `externals/llvm-project/build/bin/mlir-tblgen`, or an installed
LLVM's `bin/`).

---

## 1. Concepts: what tablegen is

Nobody writing MLIR passes hand-implements the pass interface the way
Tutorial 3 §4 showed. They write a few lines in a **tablegen** file and let
a generator produce the boilerplate, keeping only `runOnOperation` (and
whatever else is genuinely custom) in C++.

Tablegen is LLVM's domain-specific language for *structured records*. A
`.td` file declares records with typed fields; a backend of the
`mlir-tblgen` tool walks those records and prints C++ code (or markdown, or
anything else a backend chooses). It is used everywhere in the LLVM
universe — instruction definitions for CPU backends, clang diagnostics —
and MLIR leans on it harder than anyone: passes (this tutorial), dialects,
ops, types, and rewrite patterns (Tutorials 5+) are all declared in
tablegen.

One mindset note before we start, straight from the article's battle scars:
tablegen is **not an abstraction layer**. You cannot use it well while
ignoring what it generates — when something goes wrong, the errors surface
in the *generated* C++, and the documentation for "what fields can I set?"
is largely the upstream `.td` base files themselves. Treat it as a totally
see-through code generator: write the `.td`, then go *read the output* (we
will). Once you approach it that way, it's genuinely pleasant — the
generated code is more complete and more correct than what you'd write by
hand.

## 2. The tablegen file

Here is [`lib/Transform/Affine/Passes.td`](../lib/Transform/Affine/Passes.td)
in full — 22 lines that replace roughly a hundred hand-written ones:

```tablegen
#ifndef LIB_TRANSFORM_AFFINE_PASSES_TD_
#define LIB_TRANSFORM_AFFINE_PASSES_TD_

include "mlir/Pass/PassBase.td"

def AffineFullUnroll : Pass<"affine-full-unroll"> {
  let summary = "Fully unroll all affine loops";
  let description = [{
    Fully unroll all affine loops.
  }];
  let dependentDialects = ["mlir::affine::AffineDialect"];
}

def AffineFullUnrollPatternRewrite : Pass<"affine-full-unroll-rewrite"> {
  let summary = "Fully unroll all affine loops using the pattern rewrite engine";
  let description = [{
    Fully unroll all affine loops using the pattern rewrite engine.
  }];
  let dependentDialects = ["mlir::affine::AffineDialect"];
}

#endif  // LIB_TRANSFORM_AFFINE_PASSES_TD_
```

Field by field:

- `include "mlir/Pass/PassBase.td"` — imports the upstream `Pass` record
  class. `PassBase.td` is also the de-facto documentation for what you may
  `let`: open it in the LLVM sources when you wonder what else is settable.
  (The `#ifndef` guard works like a C header guard; `.td` files use the C
  preprocessor conventions.)
- `def Name : Pass<...>` — `def` *instantiates* a record (as opposed to
  `class`, which declares a reusable template like `Pass` itself). The
  record's name, `AffineFullUnroll`, becomes the generated class-name stem
  (`AffineFullUnrollBase`) and the macro suffix
  (`GEN_PASS_DEF_AFFINEFULLUNROLL`).
- `Pass<"affine-full-unroll">` — the template argument is the CLI flag,
  what Tutorial 3's hand-written `getArgument()` returned. `Pass` takes an
  optional second argument naming the anchor op (e.g.
  `Pass<"...", "mlir::func::FuncOp">`); omitted here, so these are generic
  passes anchored on any op — this is exactly where the repo's passes lost
  the `FuncOp` anchoring that the article's Tutorial-3-era code had.
- `let summary = ...` — the `--help` one-liner (`getDescription()` in the
  hand-written version — confusingly, tablegen's `summary` maps to the C++
  `getDescription()`).
- `let description = [{...}]` — long-form documentation; `[{` `}]` is
  tablegen's heredoc syntax for multi-line strings. Used by the doc
  generator (section 7), not by `--help`.
- `let dependentDialects = [...]` — the important one. An `MLIRContext`
  loads dialects lazily, and a pass may only create ops from dialects that
  are already loaded. Unrolling creates fresh `affine.apply` ops, so the
  pass must declare the affine dialect, and the generated code will load it
  before the pass runs. Forgetting a dependent dialect is a classic
  runtime crash ("op created with unregistered dialect") that this field
  exists to prevent.

## 3. Step: run the generator by hand

The build runs `mlir-tblgen` for you, but running it manually makes the
whole mechanism concrete. The only flags are the backend selector
(`--gen-pass-decls`), a `-name` for the pass *group*, and an include path
so tablegen can find `PassBase.td`:

```bash
# Bazel: use any LLVM checkout/install for the include path; CMake users
# can use externals/llvm-project/mlir/include. Here: an installed LLVM.
mlir-tblgen --gen-pass-decls -name Affine \
  -I /opt/homebrew/opt/llvm@20/include \
  lib/Transform/Affine/Passes.td | wc -l
```

```
306
```

Twenty-two lines in, 306 lines of C++ out. The `-name Affine` argument is
what groups the two passes: it names the umbrella registration function
`registerAffinePasses()`.

In the real build you never see this command. With Bazel,
`gentbl_cc_library` in
[`lib/Transform/Affine/BUILD`](../lib/Transform/Affine/BUILD) runs it —
recall the target from Tutorial 3 §10:

```python
gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = [
        (["-gen-pass-decls", "-name=Affine"], "Passes.h.inc"),
        (["-gen-pass-doc"], "AffinePasses.md"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Passes.td",
    deps = ["@llvm-project//mlir:OpBaseTdFiles",
            "@llvm-project//mlir:PassBaseTdFiles"],
)
```

Each `tbl_outs` pair is (flags, output file) — one `mlir-tblgen` run each.
The outputs land in `bazel-bin/lib/Transform/Affine/`, and because that
directory is on the include path, C++ can
`#include "lib/Transform/Affine/Passes.h.inc"` as if the file lived in the
source tree. The `deps` are *tablegen-file* dependencies (the upstream
`.td` files we `include`), not C++ ones.

The CMake equivalent in
[`lib/Transform/Affine/CMakeLists.txt`](../lib/Transform/Affine/CMakeLists.txt):

```cmake
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Affine)
add_public_tablegen_target(MLIRAffineFullUnrollPasses)
add_mlir_doc(Passes AffinePasses ./ -gen-pass-doc)
```

with the output landing in the build tree
(`<build>/lib/Transform/Affine/Passes.h.inc`), which is likewise on the
include path.

## 4. Reading the generated code

Pipe the section-3 command through `less` (or open the generated file from
your build tree) and you'll find it is *three* files in one, each gated by
a preprocessor macro so that consumers opt in to exactly the part they
need. All excerpts below are from the real generated file, lightly
trimmed.

### `GEN_PASS_DECL_*` — the public face

```cpp
#ifdef GEN_PASS_DECL_AFFINEFULLUNROLL
std::unique_ptr<::mlir::Pass> createAffineFullUnroll();
#undef GEN_PASS_DECL_AFFINEFULLUNROLL
#endif // GEN_PASS_DECL_AFFINEFULLUNROLL
```

Just a factory function declaration — all that other code needs in order
to *use* the pass. This is what the pass's header,
[`AffineFullUnroll.h`](../lib/Transform/Affine/AffineFullUnroll.h),
consumes; the entire header is:

```cpp
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace tutorial {

#define GEN_PASS_DECL_AFFINEFULLUNROLL
#include "lib/Transform/Affine/Passes.h.inc"

}  // namespace tutorial
}  // namespace mlir
```

Note the idiom: `#define` the gate, then `#include` the `.inc`. The
generated file `#undef`s the gate after use, so several gates can be pulled
from the same file in sequence. Note also that the `.inc` is included
*inside* `namespace mlir::tutorial` — the generated code deliberately
contains no namespace of its own, and lands in whatever namespace you
include it into. (Get this wrong and you'll enjoy some of section 8.)

### `GEN_PASS_DEF_*` — the base class

This is the replacement for Tutorial 3's hand-written `PassWrapper` class,
consumed at the top of
[`AffineFullUnroll.cpp`](../lib/Transform/Affine/AffineFullUnroll.cpp) with
`#define GEN_PASS_DEF_AFFINEFULLUNROLL`:

```cpp
namespace impl {

template <typename DerivedT>
class AffineFullUnrollBase : public ::mlir::OperationPass<> {
public:
  using Base = AffineFullUnrollBase;

  AffineFullUnrollBase()
      : ::mlir::OperationPass<>(::mlir::TypeID::get<DerivedT>()) {}

  /// Returns the command-line argument attached to this pass.
  ::llvm::StringRef getArgument() const override { return "affine-full-unroll"; }

  ::llvm::StringRef getDescription() const override {
    return "Fully unroll all affine loops";
  }

  /// Support isa/dyn_cast functionality for the derived pass class.
  static bool classof(const ::mlir::Pass *pass) {
    return pass->getTypeID() == ::mlir::TypeID::get<DerivedT>();
  }

  /// A clone method to create a copy of this pass.
  std::unique_ptr<::mlir::Pass> clonePass() const override {
    return std::make_unique<DerivedT>(*static_cast<const DerivedT *>(this));
  }

  /// Return the dialect that must be loaded in the context before this pass.
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::affine::AffineDialect>();
  }

private:
  friend std::unique_ptr<::mlir::Pass> createAffineFullUnroll() {
    return std::make_unique<DerivedT>();
  }
};
} // namespace impl
```

Everything from Tutorial 3 §4 is here, written for you, and you can now
map each `Passes.td` field to its output: the `Pass<"...">` flag became
`getArgument()`, `summary` became `getDescription()`,
`dependentDialects` became `getDependentDialects()` (there's the lazy
dialect loading from section 2). And you get things the hand-written
version skipped: `classof` enables MLIR's `isa`/`dyn_cast` on passes via a
per-class `TypeID`, and `clonePass` is required for the pass manager to
run pass instances on multiple functions safely — the boilerplate
`PassWrapper` used to hide, now merely *generated* instead of hidden.

Note the base is `::mlir::OperationPass<>` — empty template argument, the
generic anchor from section 2. And the CRTP from Tutorial 3 is still here
(`template <typename DerivedT>` + `std::make_unique<DerivedT>()` in the
friend factory): the base class must construct and clone *your* derived
class, which it can only name via the template parameter.

What's left for the human is exactly what Tutorial 3 showed:

```cpp
struct AffineFullUnroll : impl::AffineFullUnrollBase<AffineFullUnroll> {
  using AffineFullUnrollBase::AffineFullUnrollBase;

  void runOnOperation() { ... }
};
```

### `GEN_PASS_REGISTRATION` — the hookup

```cpp
#ifdef GEN_PASS_REGISTRATION

inline void registerAffineFullUnroll() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createAffineFullUnroll();
  });
}

inline void registerAffineFullUnrollPatternRewrite() { /* same shape */ }

inline void registerAffinePasses() {
  registerAffineFullUnroll();
  registerAffineFullUnrollPatternRewrite();
}
#endif // GEN_PASS_REGISTRATION
```

This is consumed by [`Passes.h`](../lib/Transform/Affine/Passes.h) (the
"whole group" header), and `registerAffinePasses()` is precisely the
function `tutorial-opt` called in Tutorial 3 §3 — mystery resolved. Each
`register*` wraps the same `createAffineFullUnroll()` factory in
`mlir::registerPass`, which is what makes the `--affine-full-unroll` flag
exist. (The generated file also contains
`registerAffineFullUnrollPass()`-style variants marked "Old registration
code, kept for temporary backwards compatibility" — generated code has
legacy baggage too.)

## 5. The migration, before and after

Put Tutorial 3 §4's hand-written class next to what this repo keeps on
disk, and the trade is clear.

Before (article's original code, ~15 lines *per pass*, plus manual
registration in `tutorial-opt.cpp`):

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
// ... and in tutorial-opt.cpp:
mlir::PassRegistration<mlir::tutorial::AffineFullUnrollPass>();
```

After: the 6-line `def` in `Passes.td`, a 3-line struct holding
`runOnOperation`, and one `registerAffinePasses()` call that never changes
as the group grows. The behavior is identical — the CLI flag, `--help`
text, and tests didn't change at all, which is also the practical test for
a refactor like this: `bazel test //...:all` before and after.

The "before" state is not hypothetical: [`tutorial/code/03/`](code/03/)
keeps the complete pre-tablegen code buildable as its own binary
(`bazel build //tutorial/code/03:tutorial-opt-03`), so you can diff it
against `lib/Transform/Affine/` file by file — this section's migration,
laid out on disk — and confirm the identical behavior yourself.

The checklist from Tutorial 3 §10 ("add a pass = pass files + build entry
+ registration call") now has a tablegen flavor: add a `def` to
`Passes.td`, write the struct with `runOnOperation` in a `.cpp`, add the
`cc_library`/`add_mlir_library` entry — and registration is already
handled if the group is registered.

## 6. Step: watch a change flow through

To feel the "see-through generator" property, make a fresh pass definition
and run tablegen on it — no build system, no C++, just text in and text
out:

```bash
cat > /tmp/extra.td <<'EOF'
include "mlir/Pass/PassBase.td"
def AffineCountLoops : Pass<"affine-count-loops"> {
  let summary = "Count affine loops";
}
EOF
mlir-tblgen --gen-pass-decls -name Affine \
  -I /opt/homebrew/opt/llvm@20/include /tmp/extra.td | grep AFFINECOUNTLOOPS
```

```
#define GEN_PASS_DECL_AFFINECOUNTLOOPS
#ifdef GEN_PASS_DECL_AFFINECOUNTLOOPS
#undef GEN_PASS_DECL_AFFINECOUNTLOOPS
#endif // GEN_PASS_DECL_AFFINECOUNTLOOPS
#ifdef GEN_PASS_DEF_AFFINECOUNTLOOPS
#undef GEN_PASS_DEF_AFFINECOUNTLOOPS
#endif // GEN_PASS_DEF_AFFINECOUNTLOOPS
```

A new `def` materializes a new family of gates, a new base class, a new
factory, and a new entry inside `registerAffinePasses()` — all from four
lines of tablegen.

## 7. What else the pass generator gives you

Two features beyond what this repo's passes use, worth knowing exist:

- **Documentation generation.** The second `tbl_outs` entry
  (`-gen-pass-doc`) renders the `summary`/`description` fields to
  markdown. For our `Passes.td` it produces:

  ```markdown
  ### `-affine-full-unroll`

  _Fully unroll all affine loops_

  Fully unroll all affine loops.
  ### `-affine-full-unroll-rewrite`
  ...
  ```

  Upstream MLIR's entire [passes documentation
  page](https://mlir.llvm.org/docs/Passes/) is generated this way — the
  `description` field you write *is* the documentation.
- **Pass options and statistics.** A `def` can declare
  `let options = [Option<...>]` to generate CLI-configurable pass
  parameters (remember `convert-math-to-funcs{convert-ctlz}` from
  Tutorial 2? `convert-ctlz` is such an option), and
  `let statistics = [...]` for counters reported with
  `--mlir-pass-statistics`. There is also `let constructor = ...` to
  supply a custom factory. `PassBase.td` documents all of these — in the
  source-as-documentation sense from section 2.

## 8. When tablegen goes wrong

The article's sharpest commentary is about tablegen's failure modes, so
let's provoke two, for calibration.

Misspell a field in the `.td` (`let summry = ...`):

```
Passes.td:7:7: error: Value 'summry' unknown!
  let summry = "Fully unroll all affine loops";
      ^
```

That one is fine — tablegen's own parser has decent errors. The pain the
article describes lives on the *C++ side* of the boundary. Forget to
implement `runOnOperation` in your struct and the compiler says (trimmed):

```
unique_ptr.h:767:30: error: allocating an object of abstract class type
      'mlir::tutorial::AffineFullUnroll'
mlir/Pass/Pass.h:179:16: note: unimplemented pure virtual method
      'runOnOperation' in 'AffineFullUnroll'
```

Modern clang does name the missing method (older toolchains were far less
helpful, hence the article's grumbling) — but notice *where* the error
points: inside `unique_ptr.h`, reached through the generated friend
factory, two layers away from anything you wrote. This is the norm for
tablegen mistakes: wrong namespace around the `#include`, a missing
`GEN_PASS_DEF` gate, a `dependentDialects` typo — each surfaces as C++
errors *in or through generated code*. The survival skill is always the
same: open the generated `.inc` in your build tree and read what is
actually there. It's short, it's commented, and it's the ground truth.

## Differences from the original article

Almost none — uniquely in this series, the repo's current state *is* this
article's end state, so the files above match the article's "after"
picture. Small notes:

- The generated code shown here comes from LLVM 20's `mlir-tblgen`;
  the article's 2023 output differs cosmetically (and ours includes the
  amusing "Old registration code, kept for temporary backwards
  compatibility" stubs).
- The article migrates `MulToAdd` in a single commit as a capstone; in the
  repo that's long done —
  [`lib/Transform/Arith/Passes.td`](../lib/Transform/Arith/Passes.td)
  holds its `def`s (including one for a much-later PDLL tutorial).
- `tutorial-opt` uses the group registration functions; the article shows
  the transitional state where `PassRegistration<>` lines were replaced
  one by one.

## Where to go next

Passes were tablegen's warm-up act. In
[Tutorial 5: Defining a New Dialect](05-defining-a-new-dialect.md),
tablegen defines an entire **dialect** — the `poly` dialect for polynomial
arithmetic, with custom types and ops (`lib/Dialect/Poly/`) — and the
generated code grows from one base class to whole op definitions with
verifiers, parsers, and printers. The read-the-generated-code habit from
section 8 becomes essential there.

**Exercises**

1. Do section 6's experiment, then extend it: give `AffineCountLoops` a
   `description` and `dependentDialects`, regenerate, and find each field's
   landing spot in the output.
2. Delete the `include "mlir/Pass/PassBase.td"` line from a copy of
   `Passes.td` and regenerate. The error —
   `error: Couldn't find class 'Pass'` — is your reminder that `Pass` is
   not a keyword, just a record class defined in the upstream `.td`.
3. Find `Passes.h.inc` in your own build tree (Bazel:
   `bazel-bin/lib/Transform/Affine/`; CMake:
   `<build>/lib/Transform/Affine/`) and diff it against what the manual
   command from section 3 prints.
4. Read [`lib/Transform/Arith/Passes.td`](../lib/Transform/Arith/Passes.td)
   and explain why `MulToAddPdll` declares `dependentDialects` while
   `MulToAdd` doesn't — section 2's rule answers it. (Hint: which pass
   creates ops from dialects that might not be loaded yet? `MulToAdd` only
   creates `arith` ops, and its input necessarily contains `arith` ops
   already.)
5. In `PassBase.td` (upstream LLVM), find the `Option` class and sketch —
   on paper only — what adding an `unroll-factor` option to
   `AffineFullUnroll` would look like. Tutorial 5's dialect work will make
   you write such things for real.
