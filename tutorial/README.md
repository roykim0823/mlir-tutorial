# Step-by-Step Tutorials

Detailed, hands-on walkthroughs of the articles listed in the top-level
[README](../README.md), adapted to the current state of this repository
(current LLVM, Bazel with Bzlmod, renamed tools, etc.). Each file is meant to
be followed at a terminal, command by command.

Tutorial numbers do not map one-to-one onto article numbers: article 1
(*Build System / Getting Started*) is covered by the build instructions and
MLIR primer in the top-level README, and article 2 is split across the first
two tutorials — basics first, testing second.

| Tutorial | Covers article | Script |
|----------|----------------|--------|
| [1. MLIR Basics and Running a Lowering](01-mlir-basics-and-running-a-lowering.md) | [2. Running and Testing a Lowering](https://jeremykun.com/2023/08/10/mlir-running-and-testing-a-lowering/) (first half) | [`01.sh`](script/01.sh) |
| [2. Testing a Lowering](02-testing-a-lowering.md) | [2. Running and Testing a Lowering](https://jeremykun.com/2023/08/10/mlir-running-and-testing-a-lowering/) (second half) | [`02.sh`](script/02.sh) |
| [3. Writing Our First Pass](03-writing-our-first-pass.md) | [3. Writing Our First Pass](https://jeremykun.com/2023/08/10/mlir-writing-our-first-pass/) | [`03.sh`](script/03.sh) |
| [4. Using Tablegen for Passes](04-using-tablegen-for-passes.md) | [4. Using Tablegen for Passes](https://jeremykun.com/2023/08/10/mlir-using-tablegen-for-passes/) | [`04.sh`](script/04.sh) |
| [5. Defining a New Dialect](05-defining-a-new-dialect.md) | [5. Defining a New Dialect](https://jeremykun.com/2023/08/21/mlir-defining-a-new-dialect/) | [`05.sh`](script/05.sh) |
| [6. Using Traits](06-using-traits.md) | [6. Using Traits](https://jeremykun.com/2023/09/07/mlir-using-traits/) | [`06.sh`](script/06.sh) |
| [7. Folders and Constant Propagation](07-folders-and-constant-propagation.md) | [7. Folders and Constant Propagation](https://jeremykun.com/2023/09/11/mlir-folders/) | [`07.sh`](script/07.sh) |
| [8. Verifiers](08-verifiers.md) | [8. Verifiers](https://jeremykun.com/2023/09/13/mlir-verifiers/) | [`08.sh`](script/08.sh) |
| [9. Canonicalizers and Declarative Rewrite Patterns](09-canonicalizers-and-drr.md) | [9. Canonicalizers and Declarative Rewrite Patterns](https://jeremykun.com/2023/09/20/mlir-canonicalizers-and-declarative-rewrite-patterns/) | [`09.sh`](script/09.sh) |
| [10. Dialect Conversion](10-dialect-conversion.md) | [10. Dialect Conversion](https://jeremykun.com/2023/10/23/mlir-dialect-conversion/) | [`10.sh`](script/10.sh) |
| [11. Lowering through LLVM](11-lowering-through-llvm.md) | [11. Lowering through LLVM](https://jeremykun.com/2023/11/01/mlir-lowering-through-llvm/) | [`11.sh`](script/11.sh) |
| [12. A Global Optimization and Dataflow Analysis](12-global-optimization-and-dataflow-analysis.md) | [12. A Global Optimization and Dataflow Analysis](https://jeremykun.com/2023/11/15/mlir-a-global-optimization-and-dataflow-analysis/) | [`12.sh`](script/12.sh) |
| [13. Defining Patterns with PDLL](13-defining-patterns-with-pdll.md) | [13. Defining Patterns with PDLL](https://www.jeremykun.com/2024/08/04/mlir-pdll/) | [`13.sh`](script/13.sh) |

## Companion code (`code/`)

Most tutorials' code *is* the repo — their listings are excerpts of
`lib/`, `tools/`, and `tests/`. But a few tutorials show code that has no
home in the main tree, because the repo has since evolved past it (the
articles migrated hand-written passes to tablegen, C++ patterns to DRR).
The [`code/`](code/) directory keeps those listings as **buildable,
runnable code**, one subdirectory per tutorial that needs it:

- [`code/03/`](code/03/) — Tutorial 3's hand-written `PassWrapper` passes
  and `PassRegistration`-style driver (binary: `tutorial-opt-03`); also
  the "before" state of Tutorial 4's migration.
- [`code/09/`](code/09/) — Tutorial 9's C++ `DifferenceOfSquares` pattern
  (binary: `tutorial-opt-09`), superseded in the main tree by the DRR
  version.

See [`code/README.md`](code/README.md) for the conventions. Scratch
`.mlir`/`.td`/`.pdll` examples that tutorials create on the fly are *not*
kept there — the companion scripts below recreate those.

## Companion scripts

Each tutorial has a one-stop script in [`script/`](script/) that replays its
runnable examples in order — useful both for re-verifying a tutorial after a
repo change and for watching all of its commands run without typing them.
Conventions:

- **Run from anywhere** — each script `cd`s to its own directory first, so
  `./tutorial/script/07.sh` and `cd tutorial/script && ./07.sh` both work.
  Commands echo as they run (`set -x`).
- Each markdown section's commands run under a **printed banner**
  (`### 6. Step: run the unrolling pass ###`), emitted by a small `step`
  helper whose `{ ...; } 2>/dev/null` wrapping keeps the banner itself out
  of the `set -x` trace — so the output reads as an index of the
  tutorial's sections. Sections whose commands are all skipped (see below)
  keep plain comments instead of a banner.
- **Prerequisites:** the upstream LLVM tools (`mlir-opt`, `FileCheck`,
  `mlir-pdll`, ...) on `$PATH`, and for tutorials 3+ a built `tutorial-opt`.
  The scripts default to the Bazel output `bazel-bin/tools/tutorial-opt`;
  point the `TUTORIAL_OPT` environment variable elsewhere (e.g. a CMake
  build's `tools/tutorial-opt`) to override.
- Some commands are **expected to fail** — the tutorials' deliberate
  verifier/legality/FileCheck failure demos. They are marked
  `# expected to FAIL: ...` and do not abort the script.
- `bazel test` / `cmake` / `llvm-lit` steps are **not executed**; they appear
  as `# skipped: ...` comments so section coverage stays visible.
- Scratch files and build artifacts go to a `mktemp -d` directory that is
  removed at the end — nothing is written into the repo.
