# Chapter 2: Testing a Lowering

This chapter picks up where
[Chapter 1](01-mlir-basics-and-running-a-lowering.md) left off: we ran the
`--convert-math-to-funcs=convert-ctlz` lowering by hand and saw its output —
now we turn that into automated tests, using the same tools every LLVM and
MLIR project uses.

**What you will learn:**

- How LLVM-style end-to-end testing with `lit` and `FileCheck` works.
- The FileCheck directive family — what `CHECK:` really matches, and how
  `CHECK-NOT`, `CHECK-LABEL`, `CHECK-SAME`, and capture variables differ
  from it.
- How to run a test's underlying commands by hand, without lit or Bazel.
- How FileCheck (the test-time verifier) and `generate-test-checks.py`
  (the authoring-time draft generator) divide the work of exhaustive tests.
- How the tests in this repository are wired into Bazel.
- How to *execute* lowered MLIR code with `mlir-runner` to test its behavior,
  not just its syntax.

**Prerequisites:** [Chapter 1](01-mlir-basics-and-running-a-lowering.md),
and a working build as described in the top-level [README.md](../README.md).

---

## 1. Concepts: lit and FileCheck

In Chapter 1 we verified the ctlz lowering by eyeballing `mlir-opt` output.
How do we lock that behavior in as a test? A compiler is an awkward thing to
unit-test: its inputs and outputs are large strings in domain-specific
languages, and what you usually want to assert is "when I run *this tool*
on *this program*, the output has *this shape*". LLVM and MLIR grew a pair
of tools for exactly that, and this repo (like every MLIR project) uses
them:

- **[`lit`](https://llvm.org/docs/CommandGuide/lit.html)** (LLVM Integrated
  Tester) discovers test files and runs the shell commands embedded in them.
- **[`FileCheck`](https://llvm.org/docs/CommandGuide/FileCheck.html)** makes
  assertions about the output of those commands.

Both live *inside* the test file, as specially formatted comments: `RUN:`
comments tell lit what commands to execute, and `CHECK` comments tell
FileCheck what the output of those commands must (and must not) contain.
The test *is* the input program, annotated.

The plan for this chapter: get the tools (section 2), learn FileCheck's
directive language (section 3), then read
and run this repo's real ctlz tests (sections 4–5), see how Bazel and CMake
automate the whole thing (sections 6–7), and finish by testing *behavior*
instead of syntax with `mlir-runner` (section 8).

## 2. Step: get the tools

Everything below runs `mlir-opt` and `FileCheck` directly, so first build
them (plus `mlir-runner` for section 8 — one-time cost):

```bash
bazel build @llvm-project//mlir:mlir-opt \
            @llvm-project//llvm:FileCheck \
            @llvm-project//mlir:mlir-runner
```

The binaries land under `bazel-bin/external/`, but the directory names are
mangled by Bzlmod — so instead of hard-coding paths, ask Bazel where they
are and stash them in shell variables:

```bash
MLIR_OPT=$(bazel cquery --output=files @llvm-project//mlir:mlir-opt)
FILECHECK=$(bazel cquery --output=files @llvm-project//llvm:FileCheck)
MLIR_RUNNER=$(bazel cquery --output=files @llvm-project//mlir:mlir-runner)
```

> **CMake users:** skip the above — the tools live in the `bin/` directory
> of whatever LLVM build you configured against. Set the same three
> variables and every later command in this chapter works verbatim:
>
> ```bash
> # if you built the LLVM submodule per the README:
> LLVM_BIN=$PWD/externals/llvm-project/build/bin
> # or, if you configured against an installed LLVM (e.g. Homebrew):
> #LLVM_BIN=$(brew --prefix llvm)/bin
>
> MLIR_OPT=$LLVM_BIN/mlir-opt
> FILECHECK=$LLVM_BIN/FileCheck
> MLIR_RUNNER=$LLVM_BIN/mlir-runner
> ```

Run every command in this chapter from the repo root (the Bazel variables
above may hold repo-relative paths).

## 3. A FileCheck primer

FileCheck's job is simple to state: it reads a program's output on
**stdin**, reads *check directives* from a file you name, and exits 0 if
every directive is satisfied, 1 otherwise. It is **silent on success** —
the exit code is the entire interface, which is what makes it composable
into test harnesses. All the subtlety is in the directives. Every
directive is a prefix plus an optional suffix — `CHECK:`, `CHECK-NOT:`,
`CHECK-LABEL:`, `CHECK-SAME:` — and the suffix changes **where the
pattern is allowed to match**.

The precise rules for a plain `CHECK:`:

- The pattern only needs to match a **substring of one line, anywhere on
  the line** — `CHECK: arith.addi` matches the line
  `%0 = arith.addi %arg0, %arg0 : i32`.
- The pattern is (mostly) **literal text**, not a regex — but runs of
  whitespace match any whitespace, and you can embed a real regex between
  `{{` and `}}` (e.g. `{{[0-9]+}}`) or a *capture* between `[[` and `]]`
  (explained below).
- Directives must match **in order**: each one starts scanning **where
  the previous match ended** — but it may **skip any number of lines** on
  its way to a match.

That last rule is a loophole: a check will happily skip past the *end of
the thing you meant to test* and find its match somewhere else entirely.
`CHECK: func.func @square` followed by `CHECK: arith.addi` passes even
when `@square` contains no `arith.addi` — FileCheck just skips ahead
until some *later* function provides one, and a regression in `@square`
goes unnoticed. The suffixed directives exist to pin matches down:

- **`CHECK-LABEL:`** closes exactly that loophole. FileCheck first finds
  the line matching each label, uses those lines to **split the input
  into blocks**, and then processes every other directive **only within
  its own block** — a check between two labels can never match text
  beyond the next label. Labels also localize failures: without them, one
  bad check throws all the *following* checks out of alignment, and the
  error lands far from the real problem. Two rules when choosing a label:
  it should match **exactly one line** of the output — function
  signatures are the classic choice — and a label pattern **may not
  define or use capture variables** (a rule that shapes how real tests
  check signatures, as the next directive shows).
- **`CHECK-SAME:`** requires its pattern to match **on the same line as
  the previous match**, picking up where that match ended; if FileCheck
  would have to cross a newline, the check fails. Its two practical jobs:
  splitting one long assertion into readable pieces (a signature check
  becomes a label plus several `-SAME` lines), and attaching captures to
  a label (the label can't hold them; the `-SAME` directly under it can).
- **`CHECK-NOT:`** inverts the assertion: its pattern must appear
  **nowhere** between the previous match and the next positive match —
  the directive for "the lowered op is *gone*".
- **Capture variables:** `[[V:regex]]` **captures** whatever the regex
  matched into a variable named `V`, and a later `[[V]]` (no colon)
  asserts that the *same* text appears again. This is essential for SSA
  values: you can't hard-code `%0` or `%arg0` in a test, because the
  printer is free to rename values — what matters is that the value
  *defined here* is the one *used there*.

Two more mechanics before we read a real test. First, *where* directives
physically sit in the check file is irrelevant to FileCheck — it just
collects them into an ordered list, so a test can interleave its checks
with the code they describe (the repo's tests do). Second, the
`--check-prefix=NAME` flag makes FileCheck ignore `CHECK` and instead
look for directives spelled `NAME:`, `NAME-LABEL:`, `NAME-SAME:`, and so
on — that is how one file bundles several independent groups of checks
(section 8 uses it).

The family at a glance:

| Directive | The pattern must match... | Typical use |
|---|---|---|
| `CHECK:` | some line at/after the previous match; substring of the line; may skip lines | "this op appears (eventually)" |
| `CHECK-NOT:` | **nowhere** between the previous and the next positive match | "the lowered op is gone" |
| `CHECK-SAME:` | on the **same line** as the previous match | wrapped signatures; captures after a label |
| `CHECK-LABEL:` | exactly one line; splits input into independent blocks; no captures | function boundaries |
| `CHECK-NEXT:` | on the line **immediately after** the previous match | airtight sequences (not used in this repo's tests) |

(There are more — `CHECK-DAG`, `CHECK-COUNT-<n>`, ... — see the
[FileCheck documentation](https://llvm.org/docs/CommandGuide/FileCheck.html).
The next two sections put the table to work on this repo's real tests.)

## 4. Step: read a real test, and run it by hand

Now the primer pays off. A real test file is section 1's idea in the
flesh: it *is* the input program, annotated — a `RUN:` comment holding the
shell command lit will execute, and `CHECK` directives that FileCheck will
enforce against that command's output. Here is the simplest real test in
this repo, [`tests/ctlz_simple.mlir`](../tests/ctlz_simple.mlir):

***tests/ctlz_simple.mlir***
```mlir
// RUN: mlir-opt %s --convert-math-to-funcs=convert-ctlz | FileCheck %s

func.func @main(%arg0: i32) -> i32 {
  // CHECK-NOT:  math.ctlz
  // CHECK:  call
  %0 = math.ctlz %arg0 : i32
  func.return %0 : i32
}
```

The `CHECK` lines you can already read: some output line must contain
`call`, and `math.ctlz` must not appear before it — `CHECK-NOT` is the
absence directive from the table, evaluated over the window that ends at
the next positive match. Together: "after running the lowering, the ctlz
op is gone and a function call appeared." (Note this is the loose style —
plain `CHECK`s that may skip lines, nothing pinned down by labels —
deliberately so; how tight to make a test is a judgment call we return to
in section 5.)

The genuinely new line is the first one:

- `// RUN: <command>` — lit executes this shell command, with the test
  considered passing if it exits 0. `%s` is substituted with the path of
  the test file itself. So this pipes the file through the lowering pass
  and hands the result to FileCheck. (`%s` is one of a family of
  *substitutions*: `%t`, a per-test temp file, appears in section 8; the
  [lit documentation](https://llvm.org/docs/CommandGuide/lit.html#substitutions)
  has the full table; and a test suite can define its own — this repo's
  `lit.cfg.py` adds `%project_source_dir`.)
- `FileCheck %s` — the pass output arrives on stdin, and the *check file*
  is the test file itself (`%s` again). That's the idiom the primer's note
  about directive placement enables: input program and check directives
  interleaved in one file, each `CHECK` sitting next to the code it
  describes.

Before letting Bazel hide everything behind a `PASSED` line, it's worth
executing the `RUN` line yourself, exactly as lit would. lit is not magic:
it substitutes `%s` and runs the result as a shell command with the LLVM
tools on `$PATH`. Replay it with the section-2 variables:

```bash
$MLIR_OPT tests/ctlz_simple.mlir --convert-math-to-funcs=convert-ctlz \
  | $FILECHECK tests/ctlz_simple.mlir
echo $?   # prints 0
```

To see it **fail**, run the same command but *without* the lowering pass,
so `math.ctlz` survives and the `CHECK-NOT` assertion trips:

```bash
$MLIR_OPT tests/ctlz_simple.mlir | $FILECHECK tests/ctlz_simple.mlir
```

FileCheck reports the first assertion it cannot satisfy — and, perhaps
surprisingly, that is `CHECK: call`, not the `CHECK-NOT`. A `CHECK-NOT` is
only evaluated over the window ending at the next positive match; since
`call` never matches, that window never closes, and FileCheck gives up on
`call` first:

```
tests/ctlz_simple.mlir:5:12: error: CHECK: expected string not found in input
 // CHECK: call
           ^
<stdin>:1:1: note: scanning from here
module {
^
<stdin>:3:12: note: possible intended match here
 %0 = math.ctlz %arg0 : i32
           ^
```

This is FileCheck's standard error report; its anatomy is worth learning
once. The `error:` names the directive that could not be satisfied (file
and line in the *check* file); `scanning from here` points into the
*input* at where the search began; `possible intended match here` is
FileCheck's guess at what you meant — here it correctly fingers the
un-lowered `math.ctlz`. Below that (abbreviated away here) it prints the
whole input, annotated with where each directive matched or gave up.

That's the entire mechanism. Everything Bazel and lit add on top —
discovering test files, setting `$PATH`, substituting `%s` and `%t`,
reporting results — is automation around this one pipeline.

## 5. Step: exhaustive assertions, and `generate-test-checks.py`

`ctlz_simple.mlir` locks in only the barest facts. The other test file,
[`tests/ctlz.mlir`](../tests/ctlz.mlir), pins down the *entire* generated
function. Its check block begins like this:

***tests/ctlz.mlir*** (excerpt)
```mlir
// CHECK-LABEL:   func.func @main(
// CHECK-SAME:                       %[[VAL_0:.*]]: i32
// CHECK-SAME:                       ) {
// CHECK:           %[[VAL_1:.*]] = call @__mlir_math_ctlz_i32(%[[VAL_0]]) : (i32) -> i32
```

After section 3 you can read this fluently: a label on each function
signature, `-SAME` continuations carrying the argument capture (a label
itself can't capture — this is the primer's rule in the wild), and the
captured `%[[VAL_0]]` asserting that the function's argument is what gets
passed to the generated ctlz call. The dense middle of the file is captures
all the way down — note how `%[[ARG]]` and `%[[N]]`, captured many lines
earlier, are asserted to be the loop's `iter_args`:

***tests/ctlz.mlir*** (excerpt)
```mlir
// CHECK:           %[[FOR_RET:.*]]:2 = scf.for %[[I:.*]] = %[[C_1INDEX]] to %[[C_32INDEX]] step %[[C_1INDEX]]
// CHECK:               iter_args(%[[ARG_ITER:.*]] = %[[ARG]], %[[N_ITER:.*]] = %[[N]]) -> (i32, i32) {
```

You can replay this test by hand, as in section 4:

```bash
$MLIR_OPT tests/ctlz.mlir --convert-math-to-funcs=convert-ctlz \
  | $FILECHECK tests/ctlz.mlir
```

### Where do such checks come from? FileCheck vs. `generate-test-checks.py`

Nobody writes 35 lines of captures by hand. LLVM ships a helper script,
`generate-test-checks.py`, and it is worth being very clear about how it
relates to FileCheck, because the two are easy to conflate:

- **FileCheck is a verifier that runs every time the test runs.** It is a
  compiled binary in the LLVM toolchain. Input: the pass's actual output
  (stdin) + the `CHECK` directives (check file). Output: an exit code that
  decides pass/fail. It generates nothing.
- **`generate-test-checks.py` is an authoring aid that runs once, at
  test-writing time.** It is a Python script in LLVM's *source tree* (not
  installed with the toolchain). Input: one concrete output of your pass.
  Output: a **draft of the `CHECK` lines** for you to paste into the test
  file. It verifies nothing, is never executed while tests run, and no
  trace of it remains in the test — FileCheck neither knows nor cares that
  the directives were generated.

So the division of labor is: the *script* writes (a first draft of) the
assertions once; *FileCheck* enforces them forever after. The workflow:

```
run the pass  →  pipe output through generate-test-checks.py
              →  review / trim / rename the draft
              →  paste into the test file
              →  from now on, lit + FileCheck enforce it on every run
```

The script lives at `mlir/utils/generate-test-checks.py` in the LLVM
monorepo — in this repo's submodule that is
`externals/llvm-project/mlir/utils/generate-test-checks.py`. If you haven't
checked the submodule out, it's a single self-contained file you can fetch:

```bash
curl -sLo /tmp/generate-test-checks.py \
  https://raw.githubusercontent.com/llvm/llvm-project/release/20.x/mlir/utils/generate-test-checks.py
```

Feed it the lowering's output:

```bash
$MLIR_OPT tests/ctlz.mlir --convert-math-to-funcs=convert-ctlz \
  | python3 /tmp/generate-test-checks.py
```

The draft it prints (abbreviated) starts with a disclaimer about itself:

```
// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py

// The script is designed to make adding checks to
// a test case fast, it is *not* designed to be authoritative
// about what constitutes a good test! The CHECK should be
// minimized and named to reflect the test intent.

// CHECK-LABEL:   func.func @main(
// CHECK-SAME:                    %[[VAL_0:.*]]: i32) {
// CHECK:           %[[VAL_1:.*]] = call @__mlir_math_ctlz_i32(%[[VAL_0]]) : (i32) -> i32
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func private @__mlir_math_ctlz_i32(
// CHECK-SAME:                                            %[[VAL_0:.*]]: i32) -> i32 attributes {llvm.linkage = #[[?]]<linkonce_odr>} {
// CHECK:           %[[VAL_1:.*]] = arith.constant 32 : i32
...
// CHECK:               scf.yield %[[VAL_17:.*]]#0, %[[VAL_17]]#1 : i32, i32
...
```

Compare this with the checked-in [`tests/ctlz.mlir`](../tests/ctlz.mlir)
and you can see both where its checks came from and what the "review, trim,
rename" step did:

- The **shape is identical** — `CHECK-LABEL` on each function signature,
  `CHECK-SAME` for the rest of the signature, captures for every SSA value.
  The checked-in test *is* this script's output, cleaned up.
- The machine names `VAL_0, VAL_1, ...` were **renamed by hand** to
  `ARG`, `C_32`, `ARGCMP`, `FOR_RET`, ... so a human can read the
  assertions. That's what the disclaimer's "named to reflect the test
  intent" means.
- Two artifacts of this very run had to be **fixed by hand**: the script
  emitted the placeholder `#[[?]]` where it couldn't resolve the
  `#llvm.linkage<linkonce_odr>` attribute (the checked-in test spells it
  out literally), and `%[[VAL_17:.*]]#0` *defines a fresh capture at what
  is really a use* of the `scf.if` result captured earlier as `VAL_14` —
  the checked-in test correctly reuses the earlier capture as
  `%[[IF_RET]]#0`. Generated checks are a starting point, not a finished
  test.

Finally, the trade-off between the loose style (`ctlz_simple.mlir`) and the
exhaustive style (this section) is a judgment call: exhaustive checks catch
more regressions, but they break on harmless changes to the pass's output —
and because regenerating them is so cheap, there's a real temptation to
"fix" a broken test by regenerating without reading the diff, which
silently bakes a genuine regression into the expected output.

## 6. Step: how the tests are wired into Bazel

Everything above describes single files. This repo automates it so that
**every `.mlir` file in `tests/` automatically becomes a test target**. Three
pieces make that work:

**(a) [`tests/BUILD`](../tests/BUILD)** bundles the needed tools into a
`test_utilities` filegroup, then calls one macro. One rule of thumb
governs the filegroup: **every program a `RUN:` line invokes must be in
this list** — Bazel tests run sandboxed, so a tool that isn't declared
simply doesn't exist there, and the test fails with a "binary not found"
error. (This codebase's author hit exactly this when adding the runner for
section 8's functional test.) Read the `data` list with that rule in mind:

***tests/BUILD*** (excerpt)
```python
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//tests:lit.cfg.py",
        "//tests:poly_to_llvm_main.c",
        "//tools:tutorial-opt",
        "@llvm-project//clang:clang",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:count",
        "@llvm-project//llvm:llc",
        "@llvm-project//llvm:not",
        "@llvm-project//mlir:mlir-runner",
        "@llvm-project//mlir:mlir-opt",
        "@llvm-project//mlir:mlir-translate",
        "@mlir_tutorial_pip_deps//lit",
    ],
    ...
)

glob_lit_tests()
```

Entry by entry:

- **The book's future is visible in the list:** `tutorial-opt` (this
  repo's own pass driver, from Chapter 3 on), `mlir-translate`, `llc`,
  and `clang` (the path to a native executable, Chapter 11).
- **The two oddballs, `not` and `count`, are tiny LLVM test helpers:**
  `not` inverts a command's exit code, so a `RUN` line can assert that a
  command *fails* (`RUN: mlir-opt %s ... | not FileCheck %s` — "these
  checks must NOT match"); `count` checks the number of lines of output.
  Nothing in `tests/` currently uses either, but they cost nothing to
  bundle. The last entry, `@mlir_tutorial_pip_deps//lit`, is the `lit`
  Python package itself, pinned via pip (more on that below).

**(b) [`bazel/lit.bzl`](../bazel/lit.bzl)** defines that macro. For each
`.mlir` file it emits a `py_test` target that invokes `lit` on the file, with
`test_utilities` in its runfiles. In other words, each generated target is
equivalent to writing:

```python
py_test(
    name = "ctlz.mlir.test",
    srcs = ["@llvm-project//llvm:lit"],
    args = ["-v", "tests/ctlz.mlir"],
    data = [":test_utilities", ":ctlz.mlir"],
    main = "lit.py",
)
```

**(c) [`tests/lit.cfg.py`](../tests/lit.cfg.py)** configures lit itself.
This file solves the one genuinely awkward problem in the whole setup: a
`RUN:` line says just `mlir-opt`, but inside a Bazel sandbox, executables
live at exotic, build-system-dependent paths. (CMake projects solve this
with a *template* file — `lit.site.cfg.py.in` — into which the configure
step substitutes concrete paths; we'll meet that in a moment. Bazel has no
configure step, so the paths must be discovered *at runtime*.) The whole
file, minus comments:

***tests/lit.cfg.py*** (excerpt)
```python
import os
from pathlib import Path
from lit.formats import ShTest

config.name = "mlir_tutorial"
config.test_format = ShTest()
config.suffixes = [".mlir"]

runfiles_dir = Path(os.environ["RUNFILES_DIR"])

tool_relpaths = [
    "+_repo_rules+llvm-project/mlir",
    "+_repo_rules+llvm-project/llvm",
    "_main/tools",
]

config.environment["PATH"] = (
    ":".join(str(runfiles_dir.joinpath(Path(path))) for path in tool_relpaths)
    + ":"
    + os.environ["PATH"]
)

substitutions = {
    "%project_source_dir": str(runfiles_dir.joinpath(Path("_main"))),
}
config.substitutions.extend(substitutions.items())
```

Reading it top to bottom:

- The `config` object is nowhere imported — the
  [lit documentation](https://llvm.org/docs/CommandGuide/lit.html#test-suites)
  states that an instance is inserted into the module's scope when lit
  executes this file (the file's own comment calls this "odd", fairly).
- `config.suffixes = [".mlir"]` is the test-discovery rule: every `.mlir`
  file in the directory is a test. `ShTest()` means "execute the `RUN:`
  lines as shell commands".
- `RUNFILES_DIR` is set by Bazel for every test: it points at a directory
  tree containing the test's declared `data` dependencies — our
  `test_utilities` filegroup. The project's own files sit under `_main/`
  (so `tutorial-opt` is at `_main/tools/`), and each external dependency
  under its repository name — `+_repo_rules+llvm-project` is the mangled
  name Bzlmod gives the LLVM dependency.
- The `config.environment["PATH"]` assignment prepends those tool
  directories to `$PATH` — *that* is why `RUN: mlir-opt ...` works.
- The last stanza defines the custom `%project_source_dir` substitution
  mentioned in section 4 (later tests use it to reference source files like
  `tests/poly_to_llvm_main.c` by absolute path).

> **Why the pinned Python matters:** lit is a Python module, so a test
> runner that relied on the *system* Python would die with "python cannot
> find the `lit` module" unless you had run `pip install lit` yourself.
> Instead the repo pins a hermetic Python 3.13 plus `lit==18.1.8` in
> [`MODULE.bazel`](../MODULE.bazel) and [`requirements.txt`](../requirements.txt)
> — that's the `@mlir_tutorial_pip_deps//lit` entry in the filegroup above —
> so no system-Python setup is needed for the Bazel flow.

**The CMake wiring** is parallel but separate — note there are *two* lit
configs in `tests/`, one per build system:

- [`tests/CMakeLists.txt`](../tests/CMakeLists.txt) calls
  `configure_lit_site_cfg` to generate `<build-dir>/tests/lit.site.cfg.py`
  (filling in paths like the LLVM tools directory from the CMake
  configuration), and `add_lit_testsuite` to define the
  `check-mlir-tutorial` target, which runs lit over the whole suite.
- [`tests/lit.cmake.cfg.py`](../tests/lit.cmake.cfg.py) is the CMake
  counterpart of `lit.cfg.py`: it declares `.mlir` as the test suffix and
  puts the LLVM tools directory and the project's own `tools/` build
  directory (for `tutorial-opt`) on `$PATH`.

## 7. Step: run the tests

### With Bazel

Run a single test:

```bash
bazel test //tests:ctlz.mlir.test
```

Run everything in the repo:

```bash
bazel test //...:all
```

A passing run looks like:

```
//tests:ctlz.mlir.test           PASSED in 0.6s
//tests:ctlz_simple.mlir.test    PASSED in 0.5s
...
```

To see what a *failure* looks like through Bazel, try breaking an assertion:
edit `tests/ctlz_simple.mlir`, change `// CHECK: call` to
`// CHECK: call_nonexistent`, and rerun. lit prints the failing `RUN` command
followed by the same FileCheck error format from sections 3–4 (the `-v`
flag in `lit.bzl` ensures this is shown). Remember to undo the change.

### With CMake

Two setup notes, both consequences of the same fact — the CMake flow is
designed around the LLVM **submodule** at `externals/llvm-project`, built
per the README:

1. This repo's C++ sources use Bazel-style include paths
   (`mlir/include/mlir/...`) that resolve against the submodule's source
   tree, and they track the submodule's pinned LLVM version. So
   `tutorial-opt` — needed by most of the test suite — only builds against
   the submodule; configuring against a system LLVM of a different version
   (e.g. Homebrew's) fails with missing headers and API errors.
2. The `check-mlir-tutorial` target invokes `llvm-lit`, which an *installed*
   LLVM does not ship — the submodule build tree has it at
   `externals/llvm-project/build/bin/llvm-lit`. If yours is missing it,
   install lit with `pip install lit` and add
   `-DLLVM_EXTERNAL_LIT=$(which llvm-lit)` to the `cmake -G Ninja ...`
   configure command from the README.

Then run the whole suite (most tests need `tutorial-opt`, so build it first):

```bash
cmake --build build-ninja --target tutorial-opt
cmake --build build-ninja --target check-mlir-tutorial
```

Unlike Bazel, CMake defines no per-test targets — but you can invoke
`llvm-lit` on the configured test directory yourself and filter by name:

```bash
llvm-lit -sv build-ninja/tests --filter ctlz
```

```
-- Testing: 3 of 19 tests, 3 workers --

Testing Time: 0.14s

Total Discovered Tests: 19
  Excluded: 16 (84.21%)
  Passed  :  3 (15.79%)
```

(`-s` keeps the output terse, `-v` prints the failing RUN commands and
FileCheck errors when something breaks. If you haven't built `tutorial-opt`
yet, lit prints a `Did not find tutorial-opt` note — harmless for the ctlz
tests, which only need the upstream tools.)

## 8. Bonus step: testing behavior, not syntax, with `mlir-runner`

Everything so far asserts what the lowered code *looks like* — but not that
it *computes the right answer*. For that, MLIR ships `mlir-runner` (called
`mlir-cpu-runner` before LLVM 20): it takes MLIR that has been lowered all
the way to the `llvm` dialect, JIT-compiles it (just-in-time: translated to
machine code in memory and executed on the spot, no executable file
written), calls a function you name, and prints the result to stdout.

[`tests/ctlz_runner.mlir`](../tests/ctlz_runner.mlir) uses it to check that
ctlz of 7 is 29 (7 as an i32 is `00000000 00000000 00000000 00000111` —
29 leading zeros):

***tests/ctlz_runner.mlir*** (excerpt)
```mlir
// RUN: mlir-opt %s \
// RUN:   -pass-pipeline="builtin.module( \
// RUN:      convert-math-to-funcs{convert-ctlz}, \
// RUN:      func.func(convert-scf-to-cf,convert-arith-to-llvm), \
// RUN:      convert-func-to-llvm, \
// RUN:      convert-cf-to-llvm, \
// RUN:      reconcile-unrealized-casts)" \
// RUN: | mlir-runner -e test_7i32_to_29 -entry-point-result=i32 > %t
// RUN: FileCheck %s --check-prefix=CHECK_TEST_7i32_TO_29 < %t

func.func @test_7i32_to_29() -> i32 {
  %arg = arith.constant 7 : i32
  %0 = math.ctlz %arg : i32
  func.return %0 : i32
}
// CHECK_TEST_7i32_TO_29: 29
```

New pieces, one at a time:

**The pass pipeline.** `mlir-runner` only understands the `llvm` dialect, so
one lowering is not enough — we must chain the whole staircase down.
`-pass-pipeline` runs an explicit sequence:

1. `convert-math-to-funcs{convert-ctlz}` — the pass from Chapter 1 (`{...}`
   is the pipeline syntax for pass options): `math` → `scf`/`arith`.
2. `func.func(convert-scf-to-cf,convert-arith-to-llvm)` — these two passes
   are wrapped in `func.func(...)`, meaning they run *on each function*
   rather than on the whole module. They lower structured control flow to
   branch-based control flow (`scf` → `cf`) and arithmetic to `llvm`.
3. `convert-func-to-llvm`, `convert-cf-to-llvm` — lower the remaining `func`
   and `cf` ops into the `llvm` dialect.
4. `reconcile-unrealized-casts` — during partial lowering, MLIR inserts
   placeholder type casts between not-yet-converted ops; this cleanup pass
   removes the ones that became no-ops (and fails if any real mismatch
   remains).

The whole pipeline is wrapped in `builtin.module(...)`, anchoring it at the
top-level module. (Older MLIR releases accepted a pipeline without this
wrapper; current `mlir-opt` rejects the unwrapped syntax.)

You can watch each stage of the staircase yourself with the section-2 shell
variables. Run just the pipeline, without the runner:

```bash
$MLIR_OPT tests/ctlz_runner.mlir \
  -pass-pipeline="builtin.module( \
     convert-math-to-funcs{convert-ctlz}, \
     func.func(convert-scf-to-cf,convert-arith-to-llvm), \
     convert-func-to-llvm, \
     convert-cf-to-llvm, \
     reconcile-unrealized-casts)"
```

The result is `llvm` dialect code — recognizably close to LLVM IR. Here is
the top of it (abbreviated):

```mlir
module {
  llvm.func @test_7i32_to_29() -> i32 {
    %0 = llvm.mlir.constant(7 : i32) : i32
    %1 = llvm.mlir.constant(29 : i32) : i32
    llvm.return %1 : i32
  }
  llvm.func linkonce_odr @__mlir_math_ctlz_i32(%arg0: i32) -> i32
      attributes {sym_visibility = "private"} {
    %0 = llvm.mlir.constant(32 : i32) : i32
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.icmp "eq" %arg0, %1 : i32
    llvm.cond_br %2, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    llvm.br ^bb10(%0 : i32)
  ^bb2:  // pred: ^bb0
    // ...
    llvm.br ^bb3(%3, %arg0, %6 : i64, i32, i32)
  ^bb3(%7: i64, %8: i32, %9: i32):  // 2 preds: ^bb2, ^bb8
    %10 = llvm.icmp "slt" %7, %5 : i64
    llvm.cond_br %10, ^bb4, ^bb9
    // ... more blocks ...
  }
}
```

Two things to notice, and one surprise:

- The structured control flow is gone. Instead of `scf.if`/`scf.for`
  regions there are **basic blocks** (`^bb1`, `^bb2`, ...) connected by
  branches (`llvm.br`, `llvm.cond_br`) — the flat
  "labels and gotos" shape that real machines execute.
- Blocks take **arguments**: `^bb3(%7: i64, %8: i32, %9: i32)` receives
  values from whichever branch jumps to it. This is MLIR's cleaner
  formulation of what classical SSA does with "phi nodes" — it's how the
  loop's `iter_args` survive the lowering.
- The surprise: look at `@test_7i32_to_29`. It just returns
  `llvm.mlir.constant(29)`. Folding during the lowering pipeline computed
  `ctlz(7)` *at compile time* — the answer our runner "computes" below was
  already decided here, and the generated ctlz function is only kept alive
  by its `linkonce_odr` linkage. The test still guards real behavior (a
  broken lowering or folder would produce a wrong constant or a crash),
  but it's a good reminder to check what your pipeline actually leaves
  behind. To force the loop to run at runtime, the function would need an
  argument instead of a constant.

Try deleting passes from the end of the pipeline and rerunning to see each
intermediate mix of dialects.

**The runner invocation.** `-e test_7i32_to_29` names the function to
execute; `-entry-point-result=i32` declares its return type so the runner
knows how to print it. The output — `29` — goes to stdout. Manually:

```bash
$MLIR_OPT tests/ctlz_runner.mlir \
  -pass-pipeline="builtin.module( \
     convert-math-to-funcs{convert-ctlz}, \
     func.func(convert-scf-to-cf,convert-arith-to-llvm), \
     convert-func-to-llvm, \
     convert-cf-to-llvm, \
     reconcile-unrealized-casts)" \
  | $MLIR_RUNNER -e test_7i32_to_29 -entry-point-result=i32
```

```
29
```

This is the number the `CHECK_TEST_7i32_TO_29: 29` assertion matches
against.

**New lit/FileCheck features.**

- `%t` is a lit substitution for a per-test temporary file; the runner's
  stdout is saved there and then fed to FileCheck. (You could pipe directly,
  but a temp file makes debugging failures easier.)
- `--check-prefix=CHECK_TEST_7i32_TO_29` — the prefix mechanism from
  section 3, used here to bundle several independent RUN+CHECK
  groups into one file: the actual file has a second test,
  `test_7i64_to_61`, verifying the 64-bit variant with its own prefix.
- Multiple `RUN:` lines in one file are all executed; a trailing `\` splits
  one long command across lines.

Run it:

```bash
bazel test //tests:ctlz_runner.mlir.test          # Bazel
llvm-lit -sv build-ninja/tests --filter ctlz_runner   # CMake
```

This is a *functional* test: if someone changed the lowering to generate
subtly wrong loop bounds, the syntactic tests from sections 4–5 might still
pass, but this one would fail with `28` (or garbage) instead of `29`.

## Where to go next

You have now seen the full workflow this book relies on: write
MLIR by hand, run passes on it with `mlir-opt`, and lock in behavior with
lit/FileCheck tests.
[Chapter 3: Writing Our First Pass](03-writing-our-first-pass.md) uses
exactly this workflow to build a custom pass — the `tutorial-opt` binary in
this repo is the project's own version of `mlir-opt` that carries those
custom passes.

**Exercises**

1. Copy the high-level ctlz program from Chapter 1 into a scratch file and
   run the full pipeline from section 8 on it with `mlir-opt`, but stop after
   step 2 (delete the last three passes). Look at the mix of `llvm`, `cf`,
   and `func` ops, and at the `unrealized_conversion_cast` ops that
   `reconcile-unrealized-casts` would later remove.
2. Add a third RUN+CHECK group to a copy of `ctlz_runner.mlir` testing
   `ctlz(1 : i32) = 31`, with its own `--check-prefix`.
3. Break `tests/ctlz.mlir` deliberately (change a captured variable like
   `%[[VAL_0]]` to `%[[VAL_1]]` somewhere) and read the FileCheck error to
   get comfortable with its failure output.
4. The last line of `tests/ctlz.mlir` is
   `// NOCVT-NOT: __mlir_math_ctlz_i32` — and no `RUN` line in the file
   ever passes `--check-prefix=NOCVT`. FileCheck only looks for the
   prefixes it is told about, so this assertion has *never been checked*
   (it has been dead since the file was written). Bring it to life:
   add a second `RUN` line that runs `mlir-opt` **without**
   `convert-ctlz` (i.e. `--convert-math-to-funcs` alone) and pipes into
   `FileCheck %s --check-prefix=NOCVT` — asserting that without the
   option, no ctlz helper function is generated. Verify it passes, then
   convince yourself it really guards something by pointing the NOCVT
   FileCheck at the *converted* output instead — you should see
   `error: NOCVT-NOT: excluded string found in input`. The lesson
   generalizes: a typo in a check prefix doesn't fail a test, it silently
   disables it.
