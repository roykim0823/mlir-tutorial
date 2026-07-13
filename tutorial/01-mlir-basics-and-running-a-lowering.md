# Chapter 1: MLIR Basics and Running a Lowering

This chapter is where the book begins: what MLIR programs look like, how to
read them, and how to run your first pass with `mlir-opt`. Everything the
later chapters build — a custom dialect, its optimizations, its descent to
native code — starts from the reading skills and the one tool introduced
here. The natural companion skill, testing, is the subject of
[Chapter 2](02-testing-a-lowering.md).

**What you will learn:**

- What *dialects* and *lowerings* are, and why MLIR is built around them.
- How to read MLIR's textual syntax (SSA values, types, regions, loops).
- How to run an existing MLIR pass with `mlir-opt`.

**Prerequisites:** a working build as described in the top-level
[README.md](../README.md). The Bazel commands below will build the tools they
need on first use — the very first invocation compiles a large chunk of
LLVM/MLIR and can take a long time (an hour or more). Subsequent runs are
cached and fast.

---

## 1. Concepts: dialects and lowerings

A traditional compiler like LLVM has essentially **one** intermediate
representation (IR): the input language is translated into LLVM IR, all
optimizations happen there, and then machine code is emitted.

MLIR ("Multi-Level IR") instead splits compilation into many smaller steps
between many small IRs. Its two central concepts:

- A **dialect** is a self-contained set of operations (and possibly types)
  with defined semantics. Dialects can sit at very different levels of
  abstraction. In this chapter you will meet:
  - [`func`](https://mlir.llvm.org/docs/Dialects/Func/) — function
    definitions, calls, and returns (`func.func`, `func.call`,
    `func.return`).
  - [`math`](https://mlir.llvm.org/docs/Dialects/MathOps/) — high-level math
    operations like `math.ctlz` ("count leading zeros").
  - [`arith`](https://mlir.llvm.org/docs/Dialects/ArithOps/) — basic
    arithmetic and comparisons (`arith.constant`, `arith.addi`, `arith.cmpi`,
    `arith.shli`).
  - [`scf`](https://mlir.llvm.org/docs/Dialects/SCFDialect/) — *structured
    control flow*: loops and conditionals that are still visible as loops and
    conditionals (`scf.for`, `scf.if`, `scf.yield`).
  - [`cf`](https://mlir.llvm.org/docs/Dialects/ControlFlowDialect/) —
    *unstructured control flow*: basic blocks and branches, one step closer
    to assembly.
  - [`llvm`](https://mlir.llvm.org/docs/Dialects/LLVM/) — an MLIR mirror of
    LLVM IR, the exit door out of MLIR.

- A **lowering** (or *conversion pass*) rewrites operations from one dialect
  into equivalent operations of lower-level dialects. Compilation in MLIR is
  a chain of lowerings: `math` → `scf`/`arith` → `cf` → `llvm` → (out of MLIR)
  → LLVM IR → machine code.

Why bother with many levels? Because some optimizations are easy at a high
level and nearly impossible at a low level. A classic example is loop
optimization: once loops have been lowered to compares and branch
instructions, an optimizer must painstakingly *re-discover* the loop
structure before it can do anything clever. MLIR's answer is to keep the
program at the right abstraction level for each optimization (e.g., the
`affine` dialect exists specifically for polyhedral loop optimizations), and
only lower when that level has nothing more to offer.

## 2. The example program

Here is a small program using the high-level `math.ctlz` operation, which
counts the leading zero bits of an integer
([`tests/ctlz_simple.mlir`](../tests/ctlz_simple.mlir) contains a version of
this):

***tests/ctlz_simple.mlir*** (excerpt)
```mlir
func.func @main(%arg0: i32) -> i32 {
  %0 = math.ctlz %arg0 : i32
  func.return %0 : i32
}
```

Some MLIR syntax notes, if this is your first time reading it:

- `func.func @main(...)` — the operation `func` from the `func` dialect
  defines a function named `@main`. Every operation is written
  `dialect.opname`.
- `%arg0`, `%0` — SSA values ("single static assignment": each value is
  assigned exactly once, ever; transformations rewire *uses* of values
  rather than mutating them).
- `: i32` — operations are explicitly typed; `i32` is a 32-bit integer.

In fact every line of MLIR has the same shape. Taking the middle line
apart:

```
   %0     =  math.ctlz   %arg0   :   i32
   └─┬─┘     └───┬────┘   └─┬─┘      └┬─┘
  result    dialect.op   operand    type
```

An operation consumes zero or more operands (SSA values) and produces zero
or more results; the trailing type annotation says what flows through. (The
[LangRef section on operations](https://mlir.llvm.org/docs/LangRef/#operations)
gives the complete spec of this structure.) There is no expression nesting
like `f(g(x))` — every intermediate result gets a name on its own line. When you meet an unfamiliar op, the
[dialect documentation](https://mlir.llvm.org/docs/Dialects/) lists every
op with its operands, results, and semantics —
[`math.ctlz`](https://mlir.llvm.org/docs/Dialects/MathOps/#mathctlz-mathcountleadingzerosop)
has its own entry.

Notice what this program does *not* say: how to actually count leading
zeros. `math.ctlz` states the intent and nothing more. Some hardware has a
dedicated instruction for it — but if we're targeting hardware that doesn't,
something must eventually spell out the algorithm in terms of simpler
operations. That is a job for a lowering, and we'll watch one do it in
step 4.

## 3. Step: run `mlir-opt` with no passes

`mlir-opt` is the command-line entry point for running passes on MLIR code
(analogous to LLVM's `opt`). Run it through Bazel on one of this repo's test
files, with no passes at all:

```bash
bazel run @llvm-project//mlir:mlir-opt -- $(pwd)/tests/ctlz_simple.mlir
```

Two Bazel quirks to know:

- Everything after `--` goes to `mlir-opt` instead of Bazel.
- The path must be **absolute** (hence `$(pwd)/...`), because Bazel runs the
  binary from its own sandboxed working directory, not from your shell's.

Output (comments from the file are stripped; ignore them):

```mlir
module {
  func.func @main(%arg0: i32) -> i32 {
    %0 = math.ctlz %arg0 : i32
    return %0 : i32
  }
}
```

Even with no passes, `mlir-opt` did three useful things:

1. **Parsed** the input (a syntax check),
2. **Verified** it (type checking and per-operation invariants),
3. **Normalized** it (wrapped everything in a top-level `module`, printed
   `return` in its shorthand form).

So `mlir-opt some_file.mlir` with no flags is a quick way to check "is this
valid MLIR?"

It's worth seeing the verifier catch something. Put this in a scratch file,
say `bad_types.mlir` (the repo also keeps a copy, `tests/wrong_type.mlir`) —
it multiplies an `i32` by an `i64`, which `arith.muli` forbids (its operands
must have matching types):

***tests/wrong_type.mlir***
```mlir
func.func @bad(%a: i32, %b: i64) -> i32 {
  %0 = arith.muli %a, %b : i32
  func.return %0 : i32
}
```

```
bad_types.mlir:2:23: error: use of value '%b' expects different type than prior uses: 'i32' vs 'i64'
  %0 = arith.muli %a, %b : i32
                      ^
bad_types.mlir:1:25: note: prior use here
func.func @bad(%a: i32, %b: i64) -> i32 {
                        ^
```

Every operation carries a source location, so errors point at the exact
line and column — you will meet these diagnostics constantly while writing
passes, and it pays to have seen a healthy one first.

One more instructive flag before we move on. The nicely readable syntax
above is per-op sugar (the "custom assembly format"). Underneath, *every*
operation has the same uniform structure, which you can display with
`--mlir-print-op-generic`:

```bash
bazel run @llvm-project//mlir:mlir-opt -- \
  --mlir-print-op-generic $(pwd)/tests/ctlz_simple.mlir
```

```mlir
"builtin.module"() ({
  "func.func"() <{function_type = (i32) -> i32, sym_name = "main"}> ({
  ^bb0(%arg0: i32):
    %0 = "math.ctlz"(%arg0) : (i32) -> i32
    "func.return"(%0) : (i32) -> ()
  }) : () -> ()
}) : () -> ()
```

Read this once and MLIR loses most of its mystery: an operation is a name
in quotes, a list of operand values in parens, a dictionary of static
*attributes* in `<{...}>` (the function's name and type are just data!),
and optionally a brace-enclosed
[*region*](https://mlir.llvm.org/docs/LangRef/#regions) of code — which is
how a "module contains functions" and a "function contains a body" are
modeled. Even `module` is an ordinary operation. The `^bb0(...)` label names
the entry [*block*](https://mlir.llvm.org/docs/LangRef/#blocks) of the
function's region — a block is a list of operations with exactly one entry
and one exit point (the classical compiler notion of a *basic block*);
blocks become important when we lower to branch-based control flow in
Chapter 2.

> **CMake users:** if you built via the CMake instructions in the README, the
> same binary is at `externals/llvm-project/build/bin/mlir-opt`, and you can
> use relative paths normally:
> `./externals/llvm-project/build/bin/mlir-opt tests/ctlz_simple.mlir`

## 4. Step: apply the lowering

MLIR has an upstream pass that converts `math` operations into function-based
implementations: `--convert-math-to-funcs`. Its `convert-ctlz` option enables
the ctlz conversion:

```bash
bazel run @llvm-project//mlir:mlir-opt -- \
  --convert-math-to-funcs=convert-ctlz $(pwd)/tests/ctlz_simple.mlir
```

Output (exact SSA value names may differ):

```mlir
module {
  func.func @main(%arg0: i32) -> i32 {
    %0 = call @__mlir_math_ctlz_i32(%arg0) : (i32) -> i32
    return %0 : i32
  }
  func.func private @__mlir_math_ctlz_i32(%arg0: i32) -> i32
      attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.cmpi eq, %arg0, %c0_i32 : i32
    %1 = scf.if %0 -> (i32) {
      scf.yield %c32_i32 : i32
    } else {
      %c1 = arith.constant 1 : index
      %c1_i32 = arith.constant 1 : i32
      %c32 = arith.constant 32 : index
      %c0_i32_0 = arith.constant 0 : i32
      %2:2 = scf.for %arg1 = %c1 to %c32 step %c1
          iter_args(%arg2 = %arg0, %arg3 = %c0_i32_0) -> (i32, i32) {
        %3 = arith.cmpi slt, %arg2, %c0_i32 : i32
        %4:2 = scf.if %3 -> (i32, i32) {
          scf.yield %arg2, %arg3 : i32, i32
        } else {
          %5 = arith.addi %arg3, %c1_i32 : i32
          %6 = arith.shli %arg2, %c1_i32 : i32
          scf.yield %6, %5 : i32, i32
        }
        scf.yield %4#0, %4#1 : i32, i32
      }
      scf.yield %2#1 : i32
    }
    return %1 : i32
  }
}
```

This is the lowering in action: **two programs, same meaning**. The
`math.ctlz` op is gone from `@main`, replaced by a call to a generated
private function `@__mlir_math_ctlz_i32` — and that function spells out the
algorithm using only the lower-level `scf` and `arith` dialects, exactly the
code we would otherwise have had to write by hand.

**The algorithm:** if the input is 0, the answer is 32; otherwise shift left
one bit at a time, counting iterations, until the top bit becomes 1 (i.e.,
the value becomes negative when read as signed).

The generated code introduces several new syntax elements worth reading
slowly:

- `index` is a distinct type from `i32`: a platform-dependent integer used
  for loop bounds and indexing (like `size_t` in C). The loop counter is an
  `index`; the values being computed are `i32`. (More details on `index` in
  [the MLIR rationale docs](https://mlir.llvm.org/docs/Rationale/Rationale/#integer-signedness-semantics).)
- `scf.for ... iter_args(...)` — loop-carried values. Since SSA values can't
  be reassigned, the loop threads its state (`%arg2`, the shifting copy of
  the input, and `%arg3`, the running count) through `iter_args`, and each
  iteration `scf.yield`s the next state.
- `%2:2 = scf.for ...` — an operation returning two results; `%2#0` and
  `%2#1` access them.
- `func.func private ... attributes {llvm.linkage = ...}` — the generated
  function is private to the module, and carries an attribute telling the
  eventual LLVM backend to deduplicate identical copies of it across
  compilation units (`linkonce_odr`), since every module that lowers a ctlz
  gets its own copy.

Every registered pass gets its own command-line flag like
`--convert-math-to-funcs`. The MLIR documentation keeps a
[complete list of passes](https://mlir.llvm.org/docs/Passes/) owned by the
upstream project. You can chain several passes with repeated flags, or use
`--pass-pipeline` for precise control (Chapter 2 needs that when lowering
all the way to executable code).

## Where to go next

You can now write MLIR by hand, check it with `mlir-opt`, and run an existing
lowering on it. The natural next question is: *how do we make sure a lowering
keeps working as the code evolves?* That is the subject of
[Chapter 2: Testing a Lowering](02-testing-a-lowering.md), which introduces
`lit` and `FileCheck` — the testing tools used by all of LLVM and by the rest
of this book.

**Exercises**

1. Write a small function of your own using `arith.muli` (integer
   multiplication) and check it with `mlir-opt`. Then break it in ways
   *different* from the section-3 demo and read each diagnostic: return an
   `i64` from a function declared `-> i32`; use a value `%x` that is never
   defined; misspell a dialect name (`arith.mulli`).
2. Walk through the generated `@__mlir_math_ctlz_i32` by hand with the input
   `7 : i32` (binary `0...0111`) and convince yourself the result is 29.
   Chapter 2 turns exactly this check into an automated test.
3. Run `mlir-opt --help | grep convert-` to see the full list of available
   conversion passes. The names alone sketch MLIR's lowering landscape.
