# MLIR for Beginners

A small, hands-on book about MLIR, built around the code in this
repository. Each chapter is meant to be followed at a terminal, command by
command: every listing is real code from this repo (or a scratch file the
chapter creates on the fly), every command has been run, and every output
shown is genuine.

The book takes one project from nothing to a working compiler: a custom
`poly` dialect for polynomial arithmetic — its syntax, semantics,
verification, optimization, and finally its lowering through LLVM to a
native executable a C program can call — plus a second dialect (`noisy`)
built to demonstrate program analysis and a solver-backed global
optimization, and a tour of MLIR's declarative pattern languages.

**Prerequisites:** a working build as described in the top-level
[README](../README.md), which also contains a short "What is MLIR?" primer
and the build-system tour (Bazel primary, CMake supported). Chapters 1–2
need only upstream MLIR tools; Chapter 3 onward also uses this repo's
`tutorial-opt` binary.

## Chapters

| Chapter | Script |
|---------|--------|
| [1. MLIR Basics and Running a Lowering](01-mlir-basics-and-running-a-lowering.md) | [`01.sh`](script/01.sh) |
| [2. Testing a Lowering](02-testing-a-lowering.md) | [`02.sh`](script/02.sh) |
| [3. Writing Our First Pass](03-writing-our-first-pass.md) | [`03.sh`](script/03.sh) |
| [4. Using Tablegen for Passes](04-using-tablegen-for-passes.md) | [`04.sh`](script/04.sh) |
| [5. Defining a New Dialect](05-defining-a-new-dialect.md) | [`05.sh`](script/05.sh) |
| [6. Using Traits](06-using-traits.md) | [`06.sh`](script/06.sh) |
| [7. Folders and Constant Propagation](07-folders-and-constant-propagation.md) | [`07.sh`](script/07.sh) |
| [8. Verifiers](08-verifiers.md) | [`08.sh`](script/08.sh) |
| [9. Canonicalizers and Declarative Rewrite Patterns](09-canonicalizers-and-drr.md) | [`09.sh`](script/09.sh) |
| [10. Dialect Conversion](10-dialect-conversion.md) | [`10.sh`](script/10.sh) |
| [11. Lowering through LLVM](11-lowering-through-llvm.md) | [`11.sh`](script/11.sh) |
| [12. A Global Optimization and Dataflow Analysis](12-global-optimization-and-dataflow-analysis.md) | [`12.sh`](script/12.sh) |
| [13. Defining Patterns with PDLL](13-defining-patterns-with-pdll.md) | [`13.sh`](script/13.sh) |

Chapters build on each other — each opens with its prerequisites and closes
with a "Where to go next" pointer — but chapters 1–4 (core MLIR machinery),
5–9 (building a dialect), and 10–13 (lowering, analysis, and pattern
languages) also work as self-contained arcs if you already know the
earlier material.

## Companion code (`code/`)

Most chapters' code *is* the repo — their listings are excerpts of `lib/`,
`tools/`, and `tests/`. But a few chapters teach an implementation style
the main tree has since evolved past (hand-written passes were migrated to
tablegen, C++ patterns to DRR). The [`code/`](code/) directory keeps those
teaching versions as **buildable, runnable code**, one subdirectory per
chapter that needs it:

- [`code/03/`](code/03/) — Chapter 3's hand-written `PassWrapper` passes
  and `PassRegistration`-style driver (binary: `tutorial-opt-03`); also
  the "before" state of Chapter 4's migration.
- [`code/09/`](code/09/) — Chapter 9's C++ `DifferenceOfSquares` pattern
  (binary: `tutorial-opt-09`), superseded in the main tree by the DRR
  version.

See [`code/README.md`](code/README.md) for the conventions. Scratch
`.mlir`/`.td`/`.pdll` examples that chapters create on the fly are *not*
kept there — the companion scripts below recreate those.

## Companion scripts

Each chapter has a one-stop script in [`script/`](script/) that replays its
runnable examples in order — useful both for re-verifying a chapter after a
repo change and for watching all of its commands run without typing them.
Conventions:

- **Run from anywhere** — each script `cd`s to its own directory first, so
  `./tutorial/script/07.sh` and `cd tutorial/script && ./07.sh` both work.
  Commands echo as they run (`set -x`).
- Each markdown section's commands run under a **printed banner**
  (`### 6. Step: run the unrolling pass ###`), emitted by a small `step`
  helper whose `{ ...; } 2>/dev/null` wrapping keeps the banner itself out
  of the `set -x` trace — so the output reads as an index of the
  chapter's sections. Sections whose commands are all skipped (see below)
  keep plain comments instead of a banner.
- **Prerequisites:** the upstream LLVM tools (`mlir-opt`, `FileCheck`,
  `mlir-pdll`, ...) on `$PATH`, and for chapters 3+ a built `tutorial-opt`.
  The scripts default to the Bazel output `bazel-bin/tools/tutorial-opt`;
  point the `TUTORIAL_OPT` environment variable elsewhere (e.g. a CMake
  build's `tools/tutorial-opt`) to override.
- Some commands are **expected to fail** — the chapters' deliberate
  verifier/legality/FileCheck failure demos. They are marked
  `# expected to FAIL: ...` and do not abort the script.
- `bazel test` / `cmake` / `llvm-lit` steps are **not executed**; they appear
  as `# skipped: ...` comments so section coverage stays visible.
- Scratch files and build artifacts go to a `mktemp -d` directory that is
  removed at the end — nothing is written into the repo.

## Origins and acknowledgments

This book grew out of Jeremy Kun's excellent *MLIR for Beginners* article
series (2023–2024, starting with
[*Build System (Getting Started)*](https://jeremykun.com/2023/08/10/mlir-getting-started/);
the full list is in the top-level [README](../README.md)), whose companion
repository this is. The chapters cover everything the articles do,
expanded with verified outputs and adapted to the repository's current
state (current LLVM, Bazel with Bzlmod, renamed tools); they stand alone
and no longer track the articles. The war stories credited to "this
codebase's author" throughout are his.
