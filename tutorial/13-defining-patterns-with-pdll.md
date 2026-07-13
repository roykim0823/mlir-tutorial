# Chapter 13: Defining Patterns with PDLL

This closing chapter doubles as a language tutorial built on the official
[PDLL documentation](https://mlir.llvm.org/docs/PDLL/): section 3 teaches
PDLL from scratch, hands-on, before section 4 applies it to this repo. It
is the book's epilogue: a *third* way to write rewrite patterns, after C++
([Chapter 3](03-writing-our-first-pass.md)) and DRR
([Chapter 9](09-canonicalizers-and-drr.md)), applied to an old friend —
the mul-to-add transformation from Chapter 3, reborn as
`--mul-to-add-pdll`.

PDLL is worth a whole chapter not because the syntax is nicer (though it
is), but because of a genuinely different *execution model*: PDLL
patterns compile to **IR** — the `pdl` dialect — and are *interpreted* at
runtime by a bytecode engine. Patterns become data. We will watch every
stage of that story in real output.

**What you will learn:**

- Why a third pattern language exists (what DRR structurally can't say).
- The PDLL language, built up step by step with runnable examples:
  patterns, variables and core types, operation expressions, constraints
  (three kinds), rewrites (three kinds).
- The runtime story: PDLL → `pdl` IR → `pdl_interp` matcher automaton →
  bytecode — all made visible with `mlir-pdll` and `mlir-opt`.
- The repo's `MulToAdd.pdll` line by line, its native C++ constraints,
  and the build wiring in both build systems.
- How to choose among C++, DRR, and PDLL — and PDLL's current limits.

**Prerequisites:** [Chapter 9](09-canonicalizers-and-drr.md) (DRR — the
contrast makes both languages clearer) and
[Chapter 3](03-writing-our-first-pass.md) (the original C++
`MulToAdd`, which you should have fresh in mind).

---

## 1. Why a third pattern language?

Chapter 9 ended with DRR looking pretty good — so why did the MLIR
community build a replacement (and put DRR into maintenance mode)? Because
DRR inherited tablegen's DAG-expression syntax, and several ordinary MLIR
shapes don't fit a tablegen DAG:

- **Multi-result operations.** DRR must bind all results to one name and
  address them with a `__N` suffix convention; PDLL writes
  `threeResultOp.output1` (or `.0`, `.1`, ...).
- **Variadic operands and result groups** — awkward to impossible in DRR.
- **Regions** — ops with bodies can't be matched in DRR at all.
- **Arithmetic on static values** — Chapter 3's `value / 2` had no DRR
  spelling whatsoever; recall Chapter 9's `CPred` escape hatch was
  string-pasted C++.
- **Readability at scale** — DRR constraints live in a list far from the
  thing they constrain; PDLL attaches them "directly on or next to the
  entities they apply to."

The designers also explicitly considered embedding pattern-writing in
Python or another host language and rejected it (dependency and tooling
consistency); PDLL is a purpose-built language with its own compiler
(`mlir-pdll`) and even its own LSP server.

### Patterns as IR

The deeper novelty is *how patterns run*. Compare the three generations:

| | authored as | becomes | executed as |
|---|---|---|---|
| C++ (Ch. 3) | `matchAndRewrite` | compiled object code | native code |
| DRR (Ch. 9) | tablegen DAGs | *generated* C++ | native code |
| PDLL | `.pdll` source | **`pdl` dialect IR** | **interpreted bytecode** |

PDLL patterns compile to operations in the
[`pdl` dialect](https://mlir.llvm.org/docs/Dialects/PDLOps/) (Pattern
Descriptor Language — not to be confused with PDLL, which humorously
stands for Pattern Descriptor Language *Language*) — MLIR IR that
*describes matching and rewriting MLIR IR* (ops like `pdl.operation`,
`pdl.replace` model "an operation", "a replacement"). At pass-build time
that IR is lowered (like any other dialect!) to
[`pdl_interp`](https://mlir.llvm.org/docs/Dialects/PDLInterpOps/), a
dialect of primitive match instructions, then to a compact bytecode that
a small interpreter executes inside the greedy driver. Only *native constraints*
(section 3.7) remain as C++.

Why would anyone interpret patterns instead of compiling them? Three
arguments:

- **Extensibility**: patterns-as-data can be loaded at *runtime* — a user
  can hand your compiler new rewrite rules without rebuilding it.
- **Size**: PDL bytecode is roughly 10× smaller than the equivalent
  compiled C++ pattern object code.
- **Speed, surprisingly**: all patterns are merged into one decision
  automaton (you'll *see* this in section 6), sharing common checks
  across patterns — per a 2019 MLIR talk (linked from
  [this discussion thread](https://discourse.llvm.org/t/what-is-the-benefit-of-interpreting-pdl/80331/1)),
  constructing this merged
  matcher is ~15× faster than compiling the same patterns as C++, and
  the maintainers claim matching itself is competitive or better.

## 2. Your lab bench: the `mlir-pdll` tool

Everything in section 3 is meant to be *run*, and the whole loop needs
one tool. `mlir-pdll` ships with every MLIR distribution, next to
Chapter 4's `mlir-tblgen` (`bazel-bin` after building any pdll target,
`externals/llvm-project/build/bin/`, or an installed LLVM's `bin/`). Its
interface:

```bash
mlir-pdll -x=mlir -I <path-to-mlir-includes> yourfile.pdll
```

- `-I` points at the directory containing `mlir/Dialect/...` tablegen
  files, so your `.td` includes resolve — same idea as Chapter 4 §3's
  include path (e.g. `-I externals/llvm-project/mlir/include`, or an
  installed LLVM's `include/`).
- `-x=` selects the output: **`mlir`** (the compiled `pdl` IR — the best
  way to *check* a pattern while learning), **`cpp`** (the generated C++
  used in real builds, section 7), or **`ast`** (the parsed syntax tree,
  for debugging "why doesn't this mean what I wrote").

Put each snippet below in a scratch `.pdll` file, run `-x=mlir`, and read
what comes out. All outputs shown are real, produced with LLVM 20 —
generated-code details vary slightly by LLVM version, and where the
official docs disagree with observed behavior (e.g. printed default
benefits), the outputs here are what the tools actually printed.

## 3. The PDLL language, step by step

### 3.1 The smallest possible pattern

A `.pdll` file is includes followed by definitions. Here is a complete,
compilable file — one anonymous pattern that erases every
`arith.constant` (not a *useful* rewrite; a minimal one):

```pdll
#include "mlir/Dialect/Arith/IR/ArithOps.td"

Pattern => erase op<arith.constant>;
```

Compile it (`mlir-pdll -x=mlir ...`) and PDLL shows you what it
understood (verified):

```mlir
pdl.pattern : benefit(0) {
  %0 = operands
  %1 = types
  %2 = operation "arith.constant"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
  rewrite %2 {
    erase %2
  }
}
```

Already visible: the pattern *is IR*; the match section binds
placeholders (`pdl.operands`, `pdl.types`, an `arith.constant`
operation); the rewrite region does the mutation. The `Pattern => ...`
form is the single-statement lambda shorthand; we'll use full bodies from
here on.

### 3.2 Includes: how PDLL knows your ops

That first line matters more than it looks:

- `#include "mlir/Dialect/Arith/IR/ArithOps.td"` — including a
  **tablegen** file imports its ODS wholesale: every op, attribute
  constraint, and interface defined there becomes available to PDLL *by
  name*. Chapters 4–5's op definitions are a machine-readable contract,
  and PDLL is a consumer. You can watch the import happen: run
  `-x=ast` on the file and the dump *begins* with dozens of
  `UserConstraintDecl`s (`APIntAttr`, ...) synthesized from arith's ODS
  (verified).
- `#include "other.pdll"` — plain textual inclusion of more PDLL, for
  sharing constraint libraries across pattern files.

### 3.3 Pattern anatomy and metadata

The general form:

```pdll
Pattern SomeName with benefit(10), recursion {
  // --- match section ---
  ...statements describing the input IR...

  // --- rewrite section ---
  replace ... with ...;   // or: erase ...;  or: rewrite ... with { ... };
}
```

- The name is optional (3.1's pattern had none) but names show up in the
  generated `pdl.pattern @Name` and in diagnostics — use them.
- The body splits at the first
  **[operation rewrite statement](https://mlir.llvm.org/docs/PDLL/#operation-rewrite-statements)**
  (`replace` / `erase` / `rewrite ... with {}`), which must be the last
  statement: everything before it is matching, everything inside it is
  rewriting. The two-phase discipline of Chapter 3 (inspect, *then*
  mutate), enforced grammatically.
- **`benefit(N)`** — the pattern priority you know from Chapters 3
  and 9. If omitted, it defaults to the number of ops matched (deeper
  patterns win ties).
- **`with recursion`** — opt-in for patterns that can safely apply to
  their own output; without it the driver guards against infinite
  self-application.

### 3.4 Variables and the core types

Every PDLL variable is typed by the kind of IR entity it stands for:

| PDLL type | C++ equivalent | you've met it as |
|---|---|---|
| `Attr` | `mlir::Attribute` | Chapter 7's fold values |
| `Op` / `Op<arith.muli>` | `mlir::Operation*` / that op's class | everywhere |
| `Type` / `TypeRange` | `mlir::Type` / `mlir::TypeRange` | Chapter 5 |
| `Value` / `ValueRange` | `mlir::Value` / `mlir::ValueRange` | Chapter 3 |

Variables get **bound** — attached to a piece of the matched IR — either
positionally inside a match expression (`x` in `op<...>(x: Value)` means
"whatever the first operand is, call it x") or by assignment
(`let halved: Attr = Halve(const);`). Three refinements:

```pdll
let value: Value;                  // declare, bind later
let value: [Value, HasOneUse];     // declare with a constraint list
let input: Value<someType>;        // constrain the value's MLIR type
_: Value                           // wildcard: match, bind nothing
```

That constraint-list form is worth savoring after Chapter 9: `HasOneUse`
— which DRR could only express as a `CPred` C++ string in a separate
constraint list — sits *on the variable declaration itself*.

### 3.5 Operation expressions

The op expression mirrors the *generic form* of an operation from
Chapter 1 §3 — which you now get to write by hand:

```pdll
let root = op<my_dialect.foo>(operands: ValueRange)
    {attr1 = someAttr: Attr} -> (resultTypes: TypeRange);
```

Piece by piece:

- **Name** in angle brackets. In a match section you may omit it —
  `op<>` matches *any* operation; in a rewrite section the name is
  mandatory (you can't create "some op").
- **Operands** in parens: one `ValueRange` for everything, or individual
  `Value`s/`ValueRange`s lining up with the op's ODS operand groups.
- **Attributes** in braces, `{name = var: Attr}` — matching an
  attribute binds its *payload*, exactly how Chapter 7's
  `poly.constant` will be picked apart below.
- **Result types** after `->`.
- **Results** of a bound op are accessed as `someOp.resultName` (ODS
  names) or `someOp.0`; and an `Op` variable auto-converts to its result
  `Value` where a value is expected. That conversion is what makes
  *nesting* work: in

  ```pdll
  op<arith.muli>(op<arith.constant> {value = c: Attr}, rhs: Value)
  ```

  the inner op expression stands for "the result of an arith.constant" —
  compare Chapter 3's `getDefiningOp<ConstantIntOp>()` dance: the
  backward walk through the SSA graph became *syntax*.

### 3.6 Step: grow a real pattern

Time to build one incrementally, the way you actually would. Goal:
eliminate `x + 0`. **Naive version** — match an add whose right operand
is any constant, keep only `x`:

```pdll
#include "mlir/Dialect/Arith/IR/ArithOps.td"

Pattern EliminateAddZeroWRONG {
  let root = op<arith.addi>(x: Value, op<arith.constant>);
  replace root with x;
}
```

This compiles — and is wrong: it rewrites `x + 7` to `x`. We matched the
*shape* but not the *content*. To inspect content we bind the constant's
payload attribute and constrain it — and "is this attribute zero?" is a
question about a C++ object, so we reach for a **native constraint**
(full taxonomy next section):

```pdll
#include "mlir/Dialect/Arith/IR/ArithOps.td"

Constraint IsZero(attr: Attr) [{
  return success(cast<::mlir::IntegerAttr>(attr).getValue().isZero());
}];

Pattern EliminateAddZero {
  let root = op<arith.addi>(x: Value, op<arith.constant> {value = zeroAttr: Attr});
  IsZero(zeroAttr);
  replace root with x;
}
```

Verified: this compiles to a `pdl.pattern` whose match section carries
`apply_native_constraint "IsZero"` — the C++ block will run during
matching, and a `failure()` from it simply means "this pattern doesn't
apply here" (Chapter 3's `return failure()`, relocated).

One more feature, discovered by trying (the docs are quiet on it, the
compiler is not): **repetition means equality**. Match `x − x` by
binding the same variable twice, and rewrite it to a constant zero using
an **attribute literal** — `attr<"...">` parses any textual MLIR
attribute:

```pdll
Pattern ZeroSelfSub {
  let root = op<arith.subi>(x: Value, x);
  rewrite root with {
    let zero = op<arith.constant> {value = attr<"0 : i32">};
    replace root with zero;
  };
}
```

Verified — the generated PDL matches `"arith.subi"(%0, %0)`: one
placeholder used twice, equality by construction. (DRR had the same
trick in Chapter 9's `(Poly_MulOp $x, $x)`; now you know both
spellings. Also note this pattern hardcodes `i32` — handling any integer
type properly needs the result type queried from `x`, a good exercise.)
And you may recognize `ZeroSelfSub` itself: it's the fold Chapter 9's
exercise 3 wished `poly.sub` had, sketched here for `arith`.

### 3.7 Constraints: the full menu

Three species, in increasing order of C++ involvement (the
[constraints section](https://mlir.llvm.org/docs/PDLL/#constraints-1) of
the PDLL docs has the full details of what can be constrained):

1. **PDLL-defined** — written in the language, composing match logic,
   optionally returning values (and tuples, with named elements accessed
   as `.0` or `.name`):

   ```pdll
   Constraint UsedByFooOp(value: Value) {
     op<my_dialect.foo>(value);
   }
   Constraint ExtractResult(op: Op<my_dialect.foo>) -> Value {
     return op.result;
   }
   ```

2. **Native, inline** — C++ in a `[{ ... }]` block (3.6's `IsZero`),
   compiled into the generated code; `rewriter` is in scope and the
   return type is implicitly `LogicalResult`. Chapter 9's `CPred`
   comparison: same escape hatch, but a *typed function* instead of a
   string pasted into a condition.

3. **Native, external** — declaration only; implementation registered at
   pass construction:

   ```pdll
   Constraint HasOneUse(value: Value);
   Constraint Halve(attr: Attr) -> Attr;   // constraints may return values
   ```

   ```cpp
   patterns.getPDLPatterns().registerConstraintFunction("HasOneUse", impl);
   ```

   The implementation can use a friendly typed signature
   (`LogicalResult f(PatternRewriter&, Value)`) or the interpreter's
   generic one (`LogicalResult f(PatternRewriter&, PDLResultList&,
   ArrayRef<PDLValue>)`), where `PDLValue` is a dynamically-typed box
   you `.cast<Attribute>()` out of — you'll see a real one in section 4.

Apply single-entity constraints in a variable's bracket list
(`let v: [Value, HasOneUse];`); multi-entity or value-returning ones with
call syntax (`IsZero(zeroAttr);`, `let halved = Halve(const);`).

### 3.8 Rewrites: the full menu

Symmetric story on the other side of the boundary. Inside patterns,
three statements:

```pdll
replace root with x;              // rewire uses, delete root
replace root with (a, b);         // multi-result replacement (tuple)
erase root;                       // just delete
rewrite root with { ... };        // a block of rewrite steps
```

In a `rewrite ... with {}` block, nothing is implicit — build ops with
`let`, and finish with a `replace`/`erase` (3.6's `ZeroSelfSub` did
exactly this). And mirroring constraints, there are function-like
**`Rewrite`** definitions in the same three flavors — PDLL-defined,
native-inline `[{ }]`, native-external via `registerRewriteFunction` —
with tuple returns and a lambda form:

```pdll
Rewrite Double(v: Value) -> Op {
  return op<arith.addi>(v, v);
}
Rewrite EraseOp(op: Op) => erase op;
```

(Verified: a pattern calling `Double(x)` inside its rewrite block
compiles, the helper inlined into the generated PDL.) One rule ties the
whole section together: **Constraints run while matching and may fail;
Rewrites run after the match is committed and must succeed.** Note also
that a `Rewrite` can only be *called* from the rewrite section —
the compiler enforces the phase discipline both ways.

### 3.9 Tooling: the AST view and the LSP

Two quality-of-life items before we graduate to real patterns. When a
pattern doesn't mean what you wrote, `-x=ast` shows the parse —
variables, their constraints, inline code blocks — in tree form
(verified output starts with the ODS imports of 3.2, then your
`PatternDecl`s). And PDLL has its own language server,
`mlir-pdll-lsp-server` (installed next to `mlir-pdll`), giving your
editor code completion for op names straight out of included ODS,
diagnostics as you type, and go-to-definition — same family as the
tblgen and MLIR LSPs, and set up the same way as Chapter 3 §12's
clangd.

---

## 4. The repo's patterns: MulToAdd in PDLL

Now the destination — and with section 3 behind you,
[`lib/Transform/Arith/MulToAdd.pdll`](../lib/Transform/Arith/MulToAdd.pdll)
reads like review. The transformation is Chapter 3's: expand
power-of-two multiplications by halving, peel everything else. The core
(one variant of each pattern; the file has both — see below):

***lib/Transform/Arith/MulToAdd.pdll*** (excerpt)
```pdll
#include "mlir/Dialect/Arith/IR/ArithOps.td"

Constraint IsPowerOfTwo(attr: Attr) [{
  int64_t value = cast<::mlir::IntegerAttr>(attr).getValue().getSExtValue();
  return success((value & (value - 1)) == 0);
}];

// Currently, constraints that return values must be defined in C++
Constraint Halve(atttr: Attr) -> Attr;
Constraint MinusOne(attr: Attr) -> Attr;

// Replace y = C*x with y = C/2*x + C/2*x, when C is a power of 2
Pattern PowerOfTwoExpandRhs with benefit(2) {
  let root = op<arith.muli>(op<arith.constant> {value = const: Attr}, rhs: Value);
  IsPowerOfTwo(const);
  let halved: Attr = Halve(const);

  rewrite root with {
    let newConst = op<arith.constant> {value = halved};
    let newMul = op<arith.muli>(newConst, rhs);
    let newAdd = op<arith.addi>(newMul, newMul);
    replace root with newAdd;
  };
}

// Replace y = 9*x with y = 8*x + x
Pattern PeelFromMulRhs with benefit(1) {
  let root = op<arith.muli>(lhs: Value, op<arith.constant> {value = const: Attr});
  let minusOne: Attr = MinusOne(const);

  rewrite root with {
    let newConst = op<arith.constant> {value = minusOne};
    let newMul = op<arith.muli>(lhs, newConst);
    let newAdd = op<arith.addi>(newMul, lhs);
    replace root with newAdd;
  };
}
```

You can name every construct now, piece by piece:

- an ODS include (3.2);
- a native inline constraint, `IsPowerOfTwo` (3.7 #2);
- two external constraints *with results*, `Halve` and `MinusOne`
  (3.7 #3 — and there's PDLL's sharpest current limit: *arithmetic on
  static values has no in-language spelling*, so `value / 2` must live
  in C++; adding arithmetic, boolean logic, and comparisons to PDLL
  [has an RFC](https://discourse.llvm.org/t/rfc-add-arithmetic-logical-and-comparison-expressions-into-pdll/78251));
- nested op matching with an attribute bind (3.5);
- benefits dividing labor exactly as in Chapter 3 (halving at 2 beats
  peeling at 1);
- `rewrite ... with {}` blocks building the replacement (3.8).

Hold it against Chapter 3's 35-line `matchAndRewrite`: the match is the
*shape*, no `getDefiningOp`, no null checks, constraints beside the
things they constrain.

Two things deserve stories:

**Why `Rhs` and `Lhs` variants of everything?** Chapter 3's C++ checked
only the right operand, on the stated assumption that canonicalization
moves constants rightward. This codebase's author got bitten here: that
normalization is a *canonicalization pattern* of `arith.muli`, and this
pass's driver only runs *these* patterns — nobody normalizes on its
behalf. The PDLL file owns the problem by spelling out both operand
orders. (A Chapter 6 aside: this is what `Commutative`-aware matching
would obviate.)

**An easter egg proving 3.7's point.** The repo's `Halve` declaration
misspells its parameter `atttr` — three t's — and nothing cares: for
external constraints only the *types* travel (via the documented
[mapping from PDLL types to C++ types](https://mlir.llvm.org/docs/PDLL/#native-constraint-type-translations));
parameter names are documentation. (Verified: fixing the typo and regenerating produces
byte-identical output.)

## 5. The pass: parsing patterns at runtime

The C++ side ([`MulToAddPdll.cpp`](../lib/Transform/Arith/MulToAddPdll.cpp))
has two jobs. First, the external constraint implementations — here is
`Halve`, in the generic `PDLValue` signature from 3.7:

***lib/Transform/Arith/MulToAddPdll.cpp*** (excerpt)
```cpp
LogicalResult halveImpl(PatternRewriter &rewriter, PDLResultList &results,
                        ArrayRef<PDLValue> args) {
  Attribute attr = args[0].cast<Attribute>();
  IntegerAttr cAttr = cast<IntegerAttr>(attr);
  int64_t value = cAttr.getValue().getSExtValue();
  results.push_back(rewriter.getIntegerAttr(cAttr.getType(), value / 2));
  return success();
}

void registerNativeConstraints(RewritePatternSet &patterns) {
  patterns.getPDLPatterns().registerConstraintFunction("Halve", halveImpl);
  patterns.getPDLPatterns().registerConstraintFunction("MinusOne", minusOneImpl);
}
```

Read it in execution order: the interpreter hands the constraint its
arguments as `PDLValue` boxes, so `args[0].cast<Attribute>()` unboxes
the attribute (3.7's dynamically-typed box, met for real); ordinary C++
then extracts the integer and builds the halved attribute; and because
`Halve` is a constraint *returning a value*, the result travels back by
pushing it onto `results`, with `success()` meaning the constraint held.
`registerNativeConstraints` then ties the `.pdll` declarations to these
implementations by name. (This signature was
discoverable only from upstream comments and unit tests — consider it
documented now.)

Second, the pass body, near-identical to Chapter 3's:

***lib/Transform/Arith/MulToAddPdll.cpp*** (excerpt)
```cpp
struct MulToAddPdll : impl::MulToAddPdllBase<MulToAddPdll> {
  using MulToAddPdllBase::MulToAddPdllBase;

  void runOnOperation() {
    mlir::RewritePatternSet patterns(&getContext());
    populateGeneratedPDLLPatterns(patterns);
    registerNativeConstraints(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};
```

`populateGeneratedPDLLPatterns` comes from the generated
`MulToAddPdll.h.inc`. What did the generator actually emit? Look
(verified, `mlir-pdll -x=cpp`):

```cpp
struct PowerOfTwoExpandRhs : ::mlir::PDLPatternModule {
  ...
  R"mlir(pdl.pattern @PowerOfTwoExpandRhs : benefit(2) {
  ...
  registerConstraintFunction("IsPowerOfTwo", IsPowerOfTwoPDLFn);
```

The "compiled" pattern is a **string of `pdl` IR** embedded in a C++
raw-string literal, parsed when the pass is constructed — with the
*inline* native constraints registered automatically (only the external
ones needed our manual `registerNativeConstraints`). That's why the
pass's tablegen declaration (Chapter 4 machinery, in
[`Passes.td`](../lib/Transform/Arith/Passes.td)) lists

***lib/Transform/Arith/Passes.td*** (excerpt)
```tablegen
let dependentDialects = [
  "mlir::pdl::PDLDialect",
  "mlir::pdl_interp::PDLInterpDialect",
];
```

— the pass *creates `pdl` IR* (by parsing that string), so the dialects
must be loaded: Chapter 10 §2's rule, in an unexpected costume.

## 6. Under the hood: watching a pattern become bytecode

Everything in this section is real output. First, PDLL → PDL — section
2's command on the repo's file:

```bash
mlir-pdll -x=mlir -I <llvm-include-path> lib/Transform/Arith/MulToAdd.pdll
```

```mlir
pdl.pattern @PowerOfTwoExpandRhs : benefit(2) {
  %1 = attribute
  %3 = operation "arith.constant" {"value" = %1} -> (%2 : !pdl.range<type>)
  %4 = result 0 of %3
  %5 = operand
  %7 = operation "arith.muli"(%4, %5 : !pdl.value, !pdl.value) -> ...
  apply_native_constraint "IsPowerOfTwo"(%1 : !pdl.attribute)
  %8 = apply_native_constraint "Halve"(%1 : !pdl.attribute) : !pdl.attribute
  rewrite %7 {
    %9 = operation "arith.constant" {"value" = %8}
    %11 = operation "arith.muli"(%10, %5 : !pdl.value, !pdl.value)
    %14 = operation "arith.addi"(%12, %13 : !pdl.value, !pdl.value)
    replace %7 with %14
  }
}
```

By now you can read this fluently — it's 3.1's toy output at full scale,
with the native constraints as named calls.

Second, PDL → `pdl_interp` — run the standard lowering
(`... | mlir-opt --convert-pdl-to-pdl-interp`):

```mlir
pdl_interp.func @matcher(%arg0: !pdl.operation) {
  pdl_interp.check_operation_name of %arg0 is "arith.muli" -> ^bb2, ^bb1
^bb1:  // 10 preds: ...
  pdl_interp.finalize
^bb2:
  pdl_interp.check_operand_count of %arg0 is 2 -> ^bb3, ^bb1
^bb3:
  %0 = pdl_interp.get_operand 0 of %arg0
  pdl_interp.is_not_null %0 : !pdl.value -> ^bb4, ^bb1
  ...
```

This is the **merged matcher automaton** from section 1's performance
claims, observable: all *four* patterns in the file became a *single*
`@matcher` function — one shared `check_operation_name ... "arith.muli"`
serving every pattern, one shared failure block (`^bb1`, with ten
predecessors!), branching apart only where the patterns genuinely differ
(which operand is the constant). Common work is done once per candidate
op, not once per pattern — that's what a C++ pattern set, where each
`matchAndRewrite` starts from scratch, structurally cannot do.

Finally, at `FrozenRewritePatternSet` construction (inside the pass),
this `pdl_interp` form is serialized to `PDLByteCode`, and the greedy
driver calls the interpreter's match/rewrite hooks like any other
pattern. Five representations of one idea — `.pdll` text, AST, `pdl`
IR, `pdl_interp` automaton, bytecode — and you've now seen all five.

## 7. Build integration and running it

Bazel ([`lib/Transform/Arith/BUILD`](../lib/Transform/Arith/BUILD)) —
Chapter 4's `gentbl_cc_library`, with one twist — the generator binary
is swapped:

***lib/Transform/Arith/BUILD*** (excerpt)
```python
gentbl_cc_library(
    name = "MulToAddPdllIncGen",
    tbl_outs = [(["-x=cpp"], "MulToAddPdll.h.inc")],
    tblgen = "@llvm-project//mlir:mlir-pdll",   # not mlir-tblgen!
    td_file = "MulToAdd.pdll",
    deps = ["@llvm-project//mlir:ArithDialect"],
)
```

CMake ([`CMakeLists.txt`](../lib/Transform/Arith/CMakeLists.txt)) has a
dedicated helper:

***lib/Transform/Arith/CMakeLists.txt*** (excerpt)
```cmake
add_mlir_pdll_library(MulToAddPdllIncGen
  MulToAdd.pdll
  MulToAddPdll.h.inc
)
```

Run the pass on [`tests/mul_to_add_pdll.mlir`](../tests/mul_to_add_pdll.mlir)
(a clone of Chapter 3's test with the new flag) — verified:

```bash
$TUTORIAL_OPT --mul-to-add-pdll tests/mul_to_add_pdll.mlir
```

```mlir
func.func @just_power_of_two(%arg0: i32) -> i32 {
  %0 = arith.addi %arg0, %arg0 : i32
  %1 = arith.addi %0, %0 : i32
  %2 = arith.addi %1, %1 : i32
  return %2 : i32
}
```

Byte-identical behavior to Chapter 3's `--mul-to-add` — same doubling
chain, same peel — arrived at through an interpreter instead of compiled
pattern code.

```bash
bazel test //tests:mul_to_add_pdll.mlir.test          # Bazel
llvm-lit -sv build-ninja/tests --filter mul_to_add_pdll  # CMake
```

## 8. Choosing among three languages

The book's final decision table:

| | C++ (Ch. 3) | DRR (Ch. 9) | PDLL |
|---|---|---|---|
| expressiveness | everything | DAGs, no regions/variadics/multi-result | rich matching; no static arithmetic (yet), no regions (yet) |
| escape hatch | is the escape hatch | `CPred` strings | typed native constraints/rewrites |
| debugging | debugger-friendly | read generated C++ | `-x=ast`, read `pdl`/`pdl_interp` IR |
| runtime model | compiled | compiled (generated C++) | interpreted bytecode |
| runtime-loadable patterns | no | no | possible by design |
| status | forever | **maintenance mode** | active development |

Known PDLL gaps as of this writing (RFC linked in section 4): no
in-language arithmetic/boolean logic/comparisons (hence `Halve` in C++ —
RFC exists), no region support, no dialect-conversion (type-converting,
Chapter 10-style) patterns. And the official docs still carry the
banner that designs "are not necessarily final." For new projects the
practical advice mirrors this repo's history: shapes-of-ops rewrites in
PDLL or DRR, anything clever in C++, and expect to be fluent in all
three because you'll *read* all three in the wild.

## The end of the road

Thirteen chapters ago, `bazel run mlir-opt -- --help` was an
achievement. Since then: lit and FileCheck (2), passes and rewrites (3),
tablegen (4), a dialect with types and ops (5), traits (6), folding (7),
verifiers (8), canonicalization and DRR (9), dialect conversion (10), a
native binary computing polynomial arithmetic (11), an ILP-powered
global optimizer (12), and patterns as interpreted IR (13). If you want
more, the thread continues in the real world: [HEIR](https://heir.dev/)
is the production FHE compiler this book prototyped, `poly` grew into a
full `polynomial` dialect (upstreamed to MLIR for a time, now maintained
in HEIR), and every mechanism in these
chapters — including PDLL — is in daily use there.

**Exercises**

1. Do section 3 for real: build `EliminateAddZero` step by step in a
   scratch file, checking each stage with `-x=mlir`. Then extend it with
   an `EliminateAddZeroLhs` twin (section 4 explains why you need one)
   and a `Rewrite`-based variant of `ZeroSelfSub` that works for any
   integer type, not just i32 (hint: you'll need a native rewrite that
   builds a zero attribute from `x`'s type).
2. The capstone: port Chapter 9's `DifferenceOfSquares` from DRR to
   PDLL on paper. You'll need a `HasOneUse` external constraint, the
   repetition trick from 3.6 for `op<poly.mul>(x: Value, x)`, and a
   `#include` of `PolyOps.td`. Compare all three spellings of this one
   pattern — C++ (Chapter 9 §3), DRR (§4), yours — side by side.
3. Run `mlir-pdll -x=ast` on `MulToAdd.pdll` and find (a) the imported
   arith ODS constraints at the top, (b) the four `PatternDecl`s,
   (c) how the inline `IsPowerOfTwo` C++ appears vs the bodiless
   `Halve`.
4. Count the sharing: in the `--convert-pdl-to-pdl-interp` output, how
   many `check_operation_name` checks exist for four patterns? Which
   block do all failures funnel into? Sketch what the automaton would
   look like if `PeelFromMul` matched `arith.subi` instead.
5. Design exercise from section 1's extensibility promise: sketch a
   `--load-patterns=<file.pdl>` flag for `tutorial-opt`. The pass
   already parses `pdl` IR from a string — what changes if the string
   comes from a user's file? What must you do about native constraints
   the file references, and what does that imply about which patterns
   are safely user-loadable?
