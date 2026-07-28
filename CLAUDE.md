# CLAUDE.md

The `tutorial/` directory of this repo is a **book** — "MLIR for
Beginners", 13 chapters (`NN-*.md`) with `tutorial/README.md` as the
preface. It grew out of Jeremy Kun's "MLIR For Beginners" article series
(jeremykun.com, listed in the top-level README), whose companion repo this
is, but as of the 2026-07-13 book conversion the chapters **do not
reference the articles**: the preface's "Origins and acknowledgments"
paragraph is the only place the series is mentioned or linked. The
instructions below captured the conventions as the chapters were written;
follow them for any edit, enhancement, or new chapter in this repo.

## Book status and layout

- **All 13 chapters are written and verified.** (Historical numbering:
  chapters 01–02 split article 2; 03–13 map 1:1 to articles 3–13;
  article 1's build-system content lives in the top-level README.)
- Every chapter is indexed in the preface's Chapter|Script table and
  chained via each file's "Where to go next" section. Keep both current
  when touching titles or adding files.
- Chapters are adapted to the **current repo state**. Drift from older
  code/toolchains (e.g. `mlir-cpu-runner` → `mlir-runner`; Bzlmod repo
  paths like `+_repo_rules+llvm-project`; `--pass-pipeline` requiring the
  `builtin.module(...)` wrapper; tablegen'd passes replacing hand-written
  `PassWrapper`; `SameOperandsAndResultType` replacing the earlier
  element-type trait) is recorded **in place** as short notes where the
  affected code/command appears — the former per-file "Differences from
  the original article" sections were dissolved into such notes (or
  dropped where redundant) during the book conversion.
- **Split dense material.** One learning goal per file. If a draft teaches
  two separable skills, make two files rather than one much longer than
  the rest.
- Repo fixes made during the chapter work: `tests/poly_syntax.mlir` had a
  broken `// RUN FileCheck` line (missing colon — FileCheck never ran);
  fixed and verified. `lib/Transform/Arith/MulToAdd.pdll` has a harmless
  `atttr` parameter-name typo — deliberately left, used as teaching
  material in chapter 13 §4 (external-constraint names don't travel).
  `tests/ctlz.mlir`'s last line (`// NOCVT-NOT: ...`) is dead — no RUN
  line uses that prefix, never has — deliberately left, used as
  chapter 02 exercise 4 (verified: passes with `--convert-math-to-funcs`
  alone, fails against converted output). `tests/filecheck_directives.mlir`
  (a fork-added demo file, five `--check-prefix` groups, two of them
  deliberately failing behind `not`) was **removed 2026-07-28** along with
  the directive-by-directive walkthrough it powered: the user judged the
  deep primer too much for this book and moved both (expanded) to a
  separate repo, `~/study/mlir-tooling-playground/lit-and-filecheck/`.
  Chapter 02 §3 is now a compact prose primer (no demo file, no runnable
  commands — 02.sh marks §3 with a plain comment), the §4 failure demo
  carries the error-anatomy explanation, and the old exercise 4 left with
  the file (exercise 5 → 4). With 19 `tests/*.mlir` files again, §7's
  `Total Discovered Tests: 19` lit output is correct as shown.

- **Companion code `tutorial/code/NN/`** holds buildable, runnable copies
  of chapter listings that exist nowhere in the main tree (user request,
  2026-07-09: "keep all the tutorial code, runnable if it's supposed to be
  runnable", explicitly with **no tablegen/macro method** in the frozen
  copies). Currently: `03/` (hand-written `PassWrapper` passes +
  `PassRegistration`-style driver, binary `tutorial-opt-03`; doubles as
  chapter 4's "before" state) and `09/` (the original C++
  `DifferenceOfSquares` pattern, binary `tutorial-opt-09`). Conventions in
  `tutorial/code/README.md`: one dir per chapter that needs it, own
  BUILD/CMakeLists, separately-named binary (frozen passes reuse the same
  CLI flags as `tutorial-opt`, so they can't share a binary), README
  mapping files → chapter sections and noting deviations. Scratch
  `.mlir`/`.td`/`.pdll` demos stay in the scripts (heredocs), NOT here. A
  2026-07 audit of all 13 chapters found these were the only non-repo
  listings besides deliberately non-runnable syntax fragments (chapter 13
  §3.3–3.5/3.7/3.8) — new chapters should keep that invariant: every
  runnable listing is either a repo file, a script heredoc, or a
  `tutorial/code/` file. These dirs are frozen teaching artifacts but they
  DO build in CI-visible targets, so LLVM bumps must touch them.
- **Companion scripts `tutorial/script/NN.sh`** (01–13) replay each
  chapter's runnable commands in order, one comment per markdown section.
  Each script `cd`s to its own directory first (runnable from any cwd);
  paths are relative to `tutorial/script/` (`TESTS="../../tests"`, bare
  upstream tool names from PATH, `TUTORIAL_OPT` defaulting to
  `../../bazel-bin/tools/tutorial-opt`, scratch files in `mktemp -d`,
  bazel/cmake/llvm-lit steps noted as skipped comments, `set -x`/`set +x`
  bracket, no `set -e` so failure demos continue, and a `step()` banner
  per section — `{ step "N. Title"; } 2>/dev/null` prints
  `### N. Title ###` without trace noise; skipped-only sections keep
  plain comments, no banner). Conventions are
  documented in `tutorial/README.md` §"Companion scripts". When a
  chapter's commands, section numbers, or section titles change, update
  its script to match (banners echo the headings verbatim). Note for 11.sh: Homebrew `mlir-translate` cannot parse the
  bazel-built `tutorial-opt`'s llvm-dialect output; it uses
  `../../bazel-bin/external/+_repo_rules+llvm-project/mlir/mlir-translate`.

## File structure

Follow the shape of the existing `tutorial/*.md` files (03 is the model of
the depth standard; 13 is the model for tool/language primers):

1. Title (`# Chapter N: ...`), then a book-style opening paragraph (1–3
   sentences: what the chapter does, linking the previous/next chapter
   files). No article links — see the voice rules below.
2. **What you will learn** bullet list; **Prerequisites** paragraph.
3. Numbered sections. Concept sections get plain titles ("1. Concepts:
   ..."); hands-on sections are titled "N. Step: ..." and are meant to be
   followed at a terminal, command by command.
4. Drift/version notes go **in place** (short asides where the affected
   code or command appears), not in a trailing differences section.
5. "Where to go next" — link the next chapter file and the repo code it
   will use.
6. **Exercises** — 2–5, doable with only what the chapter taught; at
   least one should foreshadow the next chapter. Any expected result an
   exercise states must itself be verified (see Verification).

Book voice (2026-07-13 conversion, user request "make tutorials chapters
of an MLIR book; disconnect from the articles"): cross-references are
"Chapter N"/"Chapter N §M"; the collection is "this book"; the word
"article" must not appear in chapters (the preface's acknowledgment is the
one exception, repo-wide). War stories and design experiences are
attributed to **"this codebase's author"** (the preface identifies him).
Never rename code/path tokens: `tutorial-opt`, `tutorial-opt-03/09`,
`$TUTORIAL_OPT`, `tutorial/code/...`, `tutorial/script/...`,
`mlir::tutorial` namespaces, repo name `mlir-tutorial`.

## Formatting conventions

- **Label file-content code blocks with the file path** on the line
  directly above the fence: `***tutorial/code/03/AffineFullUnroll.h***`
  (bold-italic, plain text, repo-relative, blank line before the label,
  none after). Append ` (excerpt)` when the block trims actual content —
  `...` elisions, omitted members/functions, cherry-picked lines, a
  fragment of a larger construct. Dropping only file boilerplate (include
  guards, `#include`s, namespace wrappers, RUN/CHECK comment lines)
  around one complete top-level construct does NOT need the suffix.
  Blocks that are shell commands, program output, generated code
  (`.h.inc`/`-gen-*` output), paraphrases, composites of several files,
  scratch files created on the fly, or illustrative sketches get NO
  label — the label asserts "this is what that repo file contains."
  When a label makes an adjacent "this code is in `<file>`:" sentence
  redundant, trim the sentence (keep it when it carries extra
  information, e.g. twin-file links). (User-established format,
  2026-07-10, tutorial 03 §4; applied across all tutorials 01–13.)

## Pedagogy rules (each came from explicit review feedback)

- **A chapter is exhaustive — never a compression of its sources.**
  Every explanation the source material gave (project organization, design
  rationale, motivations, historical asides, tooling/CI notes) must
  appear, *plus* added detail: verified outputs, current-repo adaptations,
  extra explanation of anything a novice would stumble on. Extract the
  source section by section and check coverage before calling a draft
  done. (Tutorial-3 feedback: the first draft compressed its source
  article and was judged harder to follow than the article itself. The
  chapters were originally written as supersets of the article series;
  that coverage is now baked in.)
- **When the subject has official upstream documentation, fold a
  step-by-step primer from those docs in BEFORE the repo walkthrough.**
  (Tutorial-13 feedback: the PDLL language tour from mlir.llvm.org/docs/PDLL
  precedes the repo walkthrough, built as incremental runnable
  examples, so the reader can read the repo's code when they reach it.)
- **Introduce a dialect/domain by what it *represents*, with worked
  hand-examples, before any tablegen.** (Tutorial-5 feedback: the
  coefficient-list convention `[1,2,3] ↔ 1+2x+3x²`, the two wraparound
  rules, and a tiny annotated program now open the poly chapter;
  chapter 12 opens with the noise model table for the same reason.)
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
  chapter 02 §2 (`$MLIR_OPT`, `$FILECHECK`, `$MLIR_RUNNER`,
  `$TUTORIAL_OPT`) so commands work verbatim for both build systems.
- **Give both Bazel and CMake versions** of every build/test command.
  Bazel is primary (the repo's primary build system); CMake goes in a
  subsection or `> **CMake users:**` blockquote.
- Ground examples in the repo's actual files (`tests/*.mlir`, `lib/...`,
  `tools/...`) and link them with relative paths (`../tests/ctlz.mlir`).
- Show **failure output**, not just success — deliberately broken
  assertions, verifier rejections, legality failures, with their real
  error messages. Meeting an error on purpose beats meeting it confused.
- Explain new MLIR syntax the first time it appears, right where it
  appears; back-reference the chapter that introduced a concept instead
  of re-explaining it.
- **When a chapter has frozen code in `tutorial/code/NN/`, teach that
  code first-class.** Open the chapter with a heads-up callout ("the code
  you'll read is NOT what `lib/` contains; it lives buildable in
  `tutorial/code/NN/`"), make the listings quote the frozen files, and
  push the current-repo form into a short "What the repo has now" aside
  at the END of the section that completes the taught implementation —
  kept minimal when a later chapter covers the migration. (Tutorial-3
  feedback, 2026-07-10: the repo-now subsection sat before §5's
  implementation and pre-empted it; reader asked for an up-front
  divergence note, the aside moved after §5, and minimum detail since
  tablegen is chapter 4's subject.)
- **Two-layer code explanations: general contract before the listing,
  code-anchored specifics after.** Before a listing, give the general
  machinery in prose — what kind of thing this is, the API contract, the
  moving parts (chapter 03 §7's OpRewritePattern/LogicalResult/greedy-
  driver intro is the model). After the listing, give the specifics:
  **bullets** when the details are parallel items (§7 piece-by-piece,
  §8 phase-by-phase), a **narrative in execution order** when the listing
  is one short linear code path (§5; user supplied the model wording) —
  never dissect a single code path into bullets. (Tutorial-3 feedback,
  2026-07-10, §5 then §7/§8: user confirmed the two-layer format and
  asked for it across the chapter.)
- **Unpack asserted mechanisms down to the moving parts.** When the text
  claims infrastructure *does* something (runs passes "concurrently",
  enforces isolation), show the mechanism: who schedules what, on what
  granularity, with a fan-out/join sketch; ground safety claims in the
  data structures (use-lists make "edits inside" into "writes outside");
  connect the machinery to code already shown (clone-per-thread ↔
  `PassWrapper`'s copy method) and to a flag the reader can toggle
  (`--mlir-disable-threading`). (Tutorial-3 §4 feedback, 2026-07-10: the
  parallelism/anchoring passage was asserted, not explained — the reader
  asked "how come the passes can be run concurrently?"; the rewrite added
  the thread-pool model, the not-pass-vs-pass clarification, the use-list
  mechanics, and a wired-loop sketch.)
- **Contrast look-alikes explicitly.** When a chapter uses several
  members of a directive/flag family (`CHECK`/`CHECK-LABEL`/`CHECK-SAME`)
  or two tools with adjacent roles (FileCheck vs
  `generate-test-checks.py`), spell out the differences head-on: one
  runnable demo per member with a passing *and* a failing run, a summary
  table, and — for tools — a "who runs, when, on what input" role
  contrast. (Tutorial-2 feedback: the draft named the directives and the
  script in passing; review asked for the differences to be explained
  clearly.)
- **Link official MLIR/LLVM docs inline.** Dialect names, pass names,
  LangRef terms, and doc-section pointers become mlir.llvm.org /
  llvm.org links at first natural mention (chapter 04 is the model).
  Verify each link's URL *and* `#anchor` still resolve — upstream docs
  drift (the ownership-based-deallocation section moved to its own page;
  `Dialects/PolynomialDialect/` was deleted when the dialect left
  upstream, now linked via web.archive.org with a removal note).
  (User request, 2026-07-13 — originally "carry over every doc link the
  source articles used"; applied across all chapters.)

## Verification (mandatory before documenting a command)

Run every command; paste real output (abbreviating is fine, inventing is
not). This applies to **exercise claims** as much as to commands: drafts
repeatedly stated wrong expected behavior that only running the code
exposed (a fabricated FileCheck error; a missed folder pre-empting a
pattern; a wrong first-firing verifier layer; an exercise hint about a
fold that doesn't exist). Verification also *found* content: the
constant-folded runner test in chapter 2, the `bufferization.clone`
pipeline gap in chapter 11, the repetition-equality answer in
chapter 13.

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
- **Proven full workaround** (reached parity during the chapter 3–13
  work; the complete recipe lives in the project memory file
  `cmake-build-requires-submodule.md`): header-shim CMake build of the
  repo's `lib/` against Homebrew LLVM 20, plus a small driver that
  registers Poly + Noisy dialects, all repo passes, and a `poly-to-llvm`
  pipeline with two LLVM-20 API adaptations
  (`OneShotBufferizationOptions`, `createConvertSCFToCFPass`), linked
  against the built archives + or-tools (`<build>/lib/*.a`, scip,
  `-framework CoreFoundation`, `-lz`). **All 18 repo lit tests pass with
  it**, and the chapter-11 native chain
  (`--poly-to-llvm | mlir-translate | llc | clang` → `Result: 351`) runs
  end to end. Copy the driver to `<build>/tools/tutorial-opt` so lit
  finds it.
- The repo's `build-ninja/` is stale (configured at an old repo path,
  empty `LLVM_EXTERNAL_LIT`) — do not use it; configure fresh builds in
  the session scratchpad with `-DLLVM_EXTERNAL_LIT=$(which llvm-lit)`.
- Run test subsets with `llvm-lit -sv <build>/tests --filter <name>`;
  Bazel per-test targets are `//tests:<file>.mlir.test`.

## Workflow (for a new chapter or a substantial revision)

1. If the subject has official MLIR docs, fetch them (WebFetch) and
   extract **section by section** — the extraction is the coverage
   checklist for the exhaustiveness rule and feeds the primer-first rule.
   (The historical first step, extracting the source article the same
   way, no longer applies: the articles are fully covered and the book is
   disconnected from them.)
2. Read the corresponding repo code (`lib/...`, `tests/...`,
   `tools/tutorial-opt.cpp`) — the repo is the source of truth for code
   and commands.
3. Draft per the structure above; verify every command and exercise
   claim; fix drafts to match reality and note divergences.
4. Before finishing, re-check the draft against the doc extraction:
   every section must be present (expanded, not compressed).
5. Update the preface's chapter table; cross-link from the previous
   chapter's "Where to go next"; keep back-references accurate if
   section numbers shifted; keep the companion script's banners and
   commands in sync.
6. When user feedback establishes a new convention, add it to the
   Pedagogy rules above (with the feedback source) so the next session
   inherits it.
