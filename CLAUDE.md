# CLAUDE.md

This repo is the code for the "MLIR For Beginners" article series
(jeremykun.com, linked in README.md). The `tutorial/` directory contains
step-by-step markdown walkthroughs of those articles — **the series is
complete** (tutorials 01–13, all articles covered). The instructions below
captured the conventions as the tutorials were written; follow them for any
edit, enhancement, or new tutorial-style document in this repo.

## Series status and layout

- **All 13 tutorials are written and verified.** Numbering: tutorials
  01–02 split article 2 (basics vs. testing); tutorials 03–13 map 1:1 to
  articles 3–13. Article 1 (build system) has no tutorial file — it lives
  in the top-level README (build instructions + "What is MLIR?" primer).
- Every tutorial is indexed in `tutorial/README.md` (tutorial → "Covers
  article" link) and chained via each file's "Where to go next" section.
  Keep both current when touching titles or adding files.
- Tutorials are adapted to the **current repo state**, not the 2023–24
  articles; each file's "Differences from the original article" section
  records the drift (e.g. `mlir-cpu-runner` → `mlir-runner`; Bzlmod repo
  paths like `+_repo_rules+llvm-project`; `--pass-pipeline` requiring the
  `builtin.module(...)` wrapper; tablegen'd passes replacing hand-written
  `PassWrapper`; `SameOperandsAndResultType` replacing the article-era
  element-type trait).
- **Split dense material.** One learning goal per file. If a draft teaches
  two separable skills, make two files rather than one much longer than
  the rest.
- Repo fixes made during the tutorial work: `tests/poly_syntax.mlir` had a
  broken `// RUN FileCheck` line (missing colon — FileCheck never ran);
  fixed and verified. `lib/Transform/Arith/MulToAdd.pdll` has a harmless
  `atttr` parameter-name typo — deliberately left, used as teaching
  material in tutorial 13 §4 (external-constraint names don't travel).
  `tests/ctlz.mlir`'s last line (`// NOCVT-NOT: ...`) is dead — no RUN
  line uses that prefix, never has — deliberately left, used as
  tutorial 02 exercise 5 (verified: passes with `--convert-math-to-funcs`
  alone, fails against converted output). `tests/filecheck_directives.mlir`
  is an addition of this fork (not upstream): the demo file for tutorial
  02 §3's FileCheck primer, five `--check-prefix` groups, the two
  deliberately-failing ones inverted with `not` in their RUN lines; it
  passes under lit with upstream tools only.

- **Companion code `tutorial/code/NN/`** holds buildable, runnable copies
  of tutorial listings that exist nowhere in the main tree (user request,
  2026-07-09: "keep all the tutorial code, runnable if it's supposed to be
  runnable", explicitly with **no tablegen/macro method** in the frozen
  copies). Currently: `03/` (hand-written `PassWrapper` passes +
  `PassRegistration`-style driver, binary `tutorial-opt-03`; doubles as
  tutorial 4's "before" state) and `09/` (the article's C++
  `DifferenceOfSquares` pattern, binary `tutorial-opt-09`). Conventions in
  `tutorial/code/README.md`: one dir per tutorial that needs it, own
  BUILD/CMakeLists, separately-named binary (frozen passes reuse the same
  CLI flags as `tutorial-opt`, so they can't share a binary), README
  mapping files → tutorial sections and noting deviations. Scratch
  `.mlir`/`.td`/`.pdll` demos stay in the scripts (heredocs), NOT here. A
  2026-07 audit of all 13 tutorials found these were the only non-repo
  listings besides deliberately non-runnable syntax fragments (tutorial 13
  §3.3–3.5/3.7/3.8) — new tutorials should keep that invariant: every
  runnable listing is either a repo file, a script heredoc, or a
  `tutorial/code/` file. These dirs are frozen teaching artifacts but they
  DO build in CI-visible targets, so LLVM bumps must touch them.
- **Companion scripts `tutorial/script/NN.sh`** (01–13) replay each
  tutorial's runnable commands in order, one comment per markdown section.
  Each script `cd`s to its own directory first (runnable from any cwd);
  paths are relative to `tutorial/script/` (`TESTS="../../tests"`, bare
  upstream tool names from PATH, `TUTORIAL_OPT` defaulting to
  `../../bazel-bin/tools/tutorial-opt`, scratch files in `mktemp -d`,
  bazel/cmake/llvm-lit steps noted as skipped comments, `set -x`/`set +x`
  bracket, no `set -e` so failure demos continue). Conventions are
  documented in `tutorial/README.md` §"Companion scripts". When a
  tutorial's commands or section numbers change, update its script to
  match. Note for 11.sh: Homebrew `mlir-translate` cannot parse the
  bazel-built `tutorial-opt`'s llvm-dialect output; it uses
  `../../bazel-bin/external/+_repo_rules+llvm-project/mlir/mlir-translate`.

## File structure

Follow the shape of the existing `tutorial/*.md` files (03 is the model of
the superset standard; 13 is the model for tool/language primers):

1. Title (`# Tutorial N: ...`), then an intro paragraph linking the source
   article and the previous/next tutorials.
2. **What you will learn** bullet list; **Prerequisites** paragraph.
3. Numbered sections. Concept sections get plain titles ("1. Concepts:
   ..."); hands-on sections are titled "N. Step: ..." and are meant to be
   followed at a terminal, command by command.
4. "Differences from the original article" (only if there are any).
5. "Where to go next" — link the next tutorial file and the repo code it
   will use.
6. **Exercises** — 2–5, doable with only what the tutorial taught; at
   least one should foreshadow the next tutorial. Any expected result an
   exercise states must itself be verified (see Verification).

## Pedagogy rules (each came from explicit review feedback)

- **A tutorial must be a SUPERSET of its article — never a compression.**
  Every explanation the article gives (project organization, design
  rationale, motivations, historical asides, tooling/CI notes) must
  appear, *plus* added detail: verified outputs, current-repo adaptations,
  extra explanation of anything a novice would stumble on. Extract the
  article section by section and check coverage before calling a draft
  done. (Tutorial-3 feedback: the first draft compressed the article and
  was judged harder to follow than the article itself.)
- **When the subject has official upstream documentation, fold a
  step-by-step primer from those docs in BEFORE the article content.**
  (Tutorial-13 feedback: the PDLL language tour from mlir.llvm.org/docs/PDLL
  now precedes the repo walkthrough, built as incremental runnable
  examples, so the reader can read the repo's code when they reach it.)
- **Introduce a dialect/domain by what it *represents*, with worked
  hand-examples, before any tablegen.** (Tutorial-5 feedback: the
  coefficient-list convention `[1,2,3] ↔ 1+2x+3x²`, the two wraparound
  rules, and a tiny annotated program now open the poly tutorial;
  tutorial 12 opens with the noise model table for the same reason.)
- **Explain what example IR values *mean*.** A syntax test still deserves
  a value-by-value semantic table ("%3 is the polynomial 1+2x+3x²; %4 is
  it evaluated at 7 = 162"), with an honest note when nothing is actually
  computed. (Tutorial-5 §7 feedback.)
- **Derive every non-obvious number in the text, and give independent
  checks.** When a fold/output produces `[1,4,10,12,9]`, show the
  schoolbook multiplication table, the general formula, a minimal
  compiler reproduction, *and* a spot-check that trusts neither (evaluate
  both sides at x=7). (Tutorial-7 feedback.)
- **Introduce code and output at the moment the reader produces it.**
  Never pre-show a later step's result as an up-front example.
  (Tutorial-1 feedback: the lowered ctlz program was shown twice; it was
  merged into the step that generates it.)
- **Show what automation hides.** For anything wrapped by Bazel/lit/CMake,
  also give the raw manual command. Reuse the shell-variable pattern from
  tutorial 02 §2 (`$MLIR_OPT`, `$FILECHECK`, `$MLIR_RUNNER`,
  `$TUTORIAL_OPT`) so commands work verbatim for both build systems.
- **Give both Bazel and CMake versions** of every build/test command.
  Bazel is primary (matches the articles); CMake goes in a subsection or
  `> **CMake users:**` blockquote.
- Ground examples in the repo's actual files (`tests/*.mlir`, `lib/...`,
  `tools/...`) and link them with relative paths (`../tests/ctlz.mlir`).
- Show **failure output**, not just success — deliberately broken
  assertions, verifier rejections, legality failures, with their real
  error messages. Meeting an error on purpose beats meeting it confused.
- Explain new MLIR syntax the first time it appears, right where it
  appears; back-reference the tutorial that introduced a concept instead
  of re-explaining it.
- **Contrast look-alikes explicitly.** When a tutorial uses several
  members of a directive/flag family (`CHECK`/`CHECK-LABEL`/`CHECK-SAME`)
  or two tools with adjacent roles (FileCheck vs
  `generate-test-checks.py`), spell out the differences head-on: one
  runnable demo per member with a passing *and* a failing run, a summary
  table, and — for tools — a "who runs, when, on what input" role
  contrast. (Tutorial-2 feedback: the draft named the directives and the
  script in passing; review asked for the differences to be explained
  clearly.)

## Verification (mandatory before documenting a command)

Run every command; paste real output (abbreviating is fine, inventing is
not). This applies to **exercise claims** as much as to commands: drafts
repeatedly stated wrong expected behavior that only running the code
exposed (a fabricated FileCheck error; a missed folder pre-empting a
pattern; a wrong first-firing verifier layer; an exercise hint about a
fold that doesn't exist). Verification also *found* content: the
constant-folded runner test in tutorial 2, the `bufferization.clone`
pipeline gap in tutorial 11, the repetition-equality answer in
tutorial 13.

Machine-specific notes for this checkout:

- Upstream tools that work directly: `/opt/homebrew/opt/llvm@20/bin/`
  (`mlir-opt`, `mlir-runner`, `mlir-translate`, `mlir-tblgen`,
  `mlir-pdll`, `FileCheck`, `llc`, `clang`); lit at
  `/opt/homebrew/bin/llvm-lit`.
- `tutorial-opt` **cannot** be built as-is against Homebrew LLVM: the
  sources use Bazel-style includes (`mlir/include/mlir/...`) resolved
  against the `externals/llvm-project` submodule (not checked out) and
  track its pinned LLVM version. Full builds need the submodule + LLVM
  built per the README, or `bazel build //tools:tutorial-opt` — both are
  hours on first use.
- **Proven full workaround** (reached parity during tutorials 3–13; the
  complete recipe lives in the project memory file
  `cmake-build-requires-submodule.md`): header-shim CMake build of the
  repo's `lib/` against Homebrew LLVM 20, plus a small driver that
  registers Poly + Noisy dialects, all repo passes, and a `poly-to-llvm`
  pipeline with two LLVM-20 API adaptations
  (`OneShotBufferizationOptions`, `createConvertSCFToCFPass`), linked
  against the built archives + or-tools (`<build>/lib/*.a`, scip,
  `-framework CoreFoundation`, `-lz`). **All 18 repo lit tests pass with
  it**, and the tutorial-11 native chain
  (`--poly-to-llvm | mlir-translate | llc | clang` → `Result: 351`) runs
  end to end. Copy the driver to `<build>/tools/tutorial-opt` so lit
  finds it.
- The repo's `build-ninja/` is stale (configured at an old repo path,
  empty `LLVM_EXTERNAL_LIT`) — do not use it; configure fresh builds in
  the session scratchpad with `-DLLVM_EXTERNAL_LIT=$(which llvm-lit)`.
- Run test subsets with `llvm-lit -sv <build>/tests --filter <name>`;
  Bazel per-test targets are `//tests:<file>.mlir.test`.

## Workflow (for a new tutorial or a substantial revision)

1. Fetch the article (WebFetch), extracting it **section by section** —
   the extraction is the coverage checklist for the superset rule. If the
   subject has official MLIR docs, fetch and extract those too (they feed
   the primer-first rule).
2. Read the corresponding repo code (`lib/...`, `tests/...`,
   `tools/tutorial-opt.cpp`) — the repo, not the article, is the source
   of truth for code and commands.
3. Draft per the structure above; verify every command and exercise
   claim; fix drafts to match reality and note divergences.
4. Before finishing, re-check the draft against the article/doc
   extraction: every section must be present (expanded, not compressed).
5. Update `tutorial/README.md`; cross-link from the previous tutorial's
   "Where to go next"; keep back-references accurate if section numbers
   shifted.
6. When user feedback establishes a new convention, add it to the
   Pedagogy rules above (with the feedback source) so the next session
   inherits it.
