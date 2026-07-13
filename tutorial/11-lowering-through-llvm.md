# Chapter 11: Lowering through LLVM

This is the book's payoff chapter: a program written in our own `poly`
dialect becomes a **native executable**, called from ordinary C code, and
prints the right answer. Chapter 10 took one step down the staircase
(`poly` → `tensor`/`scf`/`arith`); this chapter takes all the rest — and
explains the last unexplained piece of `tutorial-opt.cpp`, the
`--poly-to-llvm` mega-pipeline.

**What you will learn:**

- How to register a named pass *pipeline* (`PassPipelineRegistration`),
  and a practical methodology for composing one.
- Two genuinely new lowering concepts: **linalg** as the gateway for
  tensor arithmetic, and **bufferization** — the tensor-to-memref
  world-switch from values to memory.
- The complete descent to the `llvm` exit dialect, pass by pass, with
  real intermediate IR at each interesting stage.
- How MLIR hands off to LLVM proper: `mlir-translate`, `llc`, linking
  against a C `main` — and what the memref calling convention means for
  your exported functions.
- A pipeline *gap* you can hit today (unlowered `bufferization.clone`),
  reproduced and explained.

**Prerequisites:** [Chapter 10](10-dialect-conversion.md), plus Chapter
2 §6 (the ctlz version of this same journey — reread it; everything there
returns here at full scale). Tools: besides `$TUTORIAL_OPT`, this
chapter uses `mlir-translate`, `llc`, and `clang` — all in the same
`bin/` directories as Chapter 2 §2's tools.

---

## 1. Concepts: pipelines and the exit dialect

Chapter 2 §8 lowered `math.ctlz` to executable form with a five-pass
`-pass-pipeline` string (that flag's
[tiny DSL](https://mlir.llvm.org/docs/PassManagement/#textual-pass-pipeline-specification)
is documented under pass management). For `poly` the journey is ~18
passes, and typing
that string is not a lifestyle. The fix is the last feature of
`tutorial-opt.cpp` (Chapter 3 §3 showed everything else):

***tools/tutorial-opt.cpp*** (excerpt)
```cpp
mlir::PassPipelineRegistration<>(
    "poly-to-llvm", "Run passes to lower the poly dialect to LLVM",
    polyToLLVMPipelineBuilder);
```

which turns a C++ function that appends passes to an `OpPassManager` into
a single `--poly-to-llvm` flag
([pass pipeline registration](https://mlir.llvm.org/docs/PassManagement/#pass-pipeline-registration)
in the docs). The pipeline's destination is the
[`llvm` dialect](https://mlir.llvm.org/docs/Dialects/LLVM/)
— the **exit dialect**: a faithful MLIR mirror of LLVM IR, from
which a mechanical translation leaves MLIR entirely. (Other exits exist —
SPIR-V for GPU, EmitC for C source — but LLVM is the classic one.)

How do you *compose* such a pipeline? The honest method by which this
codebase's author built it — worth learning *as method* — is: start from
an empty pipeline;
look at the highest-level op remaining in your IR; add the pass that
lowers it (the upstream [passes list](https://mlir.llvm.org/docs/Passes/)
has too many lowerings to scan linearly — search it for the dialect's
name); when that pass complains, figure out what it needed *first*;
repeat until only the exit dialect remains. The pipeline below is the
fossil record of exactly that process — including the fixes from its
war stories (section 6).

## 2. The pipeline, stage by stage

A pipeline builder is exactly the kind of function section 1's
registration wraps: it receives an `OpPassManager` and appends passes
with `addPass`, in the order they will run — so reading it top to bottom
is reading the compilation strategy. Here is `polyToLLVMPipelineBuilder`
from [`tools/tutorial-opt.cpp`](../tools/tutorial-opt.cpp), grouped into
its five logical stages:

***tools/tutorial-opt.cpp*** (excerpt)
```cpp
void polyToLLVMPipelineBuilder(mlir::OpPassManager &manager) {
  // Stage 1: leave poly (Chapter 10)
  manager.addPass(mlir::tutorial::poly::createPolyToStandard());
  manager.addPass(mlir::createCanonicalizerPass());

  // Stage 2: tensor arithmetic -> linalg
  manager.addPass(mlir::createConvertElementwiseToLinalgPass());
  manager.addPass(mlir::createConvertTensorToLinalgPass());

  // Stage 3: bufferize (tensors -> memrefs), then free what we malloc
  mlir::bufferization::OneShotBufferizePassOptions bufferizationOptions;
  bufferizationOptions.bufferizeFunctionBoundaries = true;
  manager.addPass(
      mlir::bufferization::createOneShotBufferizePass(bufferizationOptions));
  mlir::bufferization::BufferDeallocationPipelineOptions deallocationOptions;
  mlir::bufferization::buildBufferDeallocationPipeline(manager,
                                                       deallocationOptions);

  // Stage 4: linalg -> loops; memref cleanup
  manager.addPass(mlir::createConvertLinalgToLoopsPass());
  manager.addPass(mlir::memref::createExpandStridedMetadataPass());

  // Stage 5: the LLVM descent (Chapter 2 §8's staircase, extended)
  manager.addPass(mlir::createSCFToControlFlowPass());
  manager.addPass(mlir::createConvertControlFlowToLLVMPass());
  manager.addPass(mlir::createArithToLLVMConversionPass());
  manager.addPass(mlir::createConvertFuncToLLVMPass());
  manager.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  manager.addPass(mlir::createReconcileUnrealizedCastsPass());

  // Cleanup (all Chapter 7/6 acquaintances)
  manager.addPass(mlir::createCanonicalizerPass());
  manager.addPass(mlir::createSCCPPass());
  manager.addPass(mlir::createCSEPass());
  manager.addPass(mlir::createSymbolDCEPass());
}
```

Stage 1 you own (Chapter 10). The cleanup quartet you know (canonicalize
and sccp are Chapter 7; cse is Chapter 6; symbol-dce deletes now-unused
private functions). Stages 2–5 need explaining, and the best way is to
watch IR move through them. Running example (degree 4 to stay readable):

> **A version note before you copy this code:** the listing compiles
> against the repo's pinned submodule LLVM. On LLVM 20 (e.g. Homebrew)
> two of these names differ — `OneShotBufferizePassOptions` is
> `OneShotBufferizationOptions`, and `createSCFToControlFlowPass` is
> `createConvertSCFToCFPass`. This chapter's outputs were produced with a
> driver using the LLVM-20 spellings of the same passes; that rename is
> also why `tutorial-opt.cpp` is the one file that won't compile against
> a system LLVM, as noted since Chapter 2's CMake caveats.

```mlir
func.func @add_ct(%arg : !poly.poly<4>) -> !poly.poly<4> {
  %0 = poly.constant dense<[1, 2, 3, 4]> : tensor<4xi32> : !poly.poly<4>
  %1 = poly.add %0, %arg : !poly.poly<4>
  return %1 : !poly.poly<4>
}
```

### Stage 2: why linalg?

After Chapter 10 our add is `arith.addi` *on tensors* — and here's the
snag:
[`convert-arith-to-llvm`](https://mlir.llvm.org/docs/Passes/#-convert-arith-to-llvm)
only handles scalar arith. LLVM has no
tensors; someone must decide what a tensor `addi` *means* in terms of
loops and memory. That someone is
**[linalg](https://mlir.llvm.org/docs/Dialects/Linalg/)**, MLIR's
structured linear-algebra dialect.
[`convert-elementwise-to-linalg`](https://mlir.llvm.org/docs/Passes/#-convert-elementwise-to-linalg)
(legal because of
`ElementwiseMappable` — Chapter 6 paying rent again) rewrites the tensor
addi into (real output):

```mlir
%0 = linalg.generic {indexing_maps = [#map, #map, #map],
                     iterator_types = ["parallel"]}
     ins(%arg0, %cst : tensor<4xi32>, tensor<4xi32>)
     outs(%arg0 : tensor<4xi32>) {
^bb0(%in: i32, %in_0: i32, %out: i32):
  %1 = arith.addi %in, %in_0 : i32
  linalg.yield %1 : i32
} -> tensor<4xi32>
```

Read `linalg.generic` as "a loop nest as data": the `indexing_maps` say
how each input/output is indexed (here identity — elementwise), the
`iterator_types` say the loop is `parallel` (no cross-iteration
dependence), and the region holds the *scalar* body. High-level enough to
optimize (fuse, tile, vectorize), mechanical enough to lower.
`convert-tensor-to-linalg` does the same for `tensor.pad` (Chapter 10's
from_tensor lowering), which becomes a `linalg.fill` plus insertion.

### Stage 3: bufferization — values become memory

Everything so far — SSA values, `tensor`s, Chapter 10's "functional
tensors" note — lives in a world of immutable *values*. Real machines
have mutable *memory*. **Bufferization** is the world-switch: every
`tensor<4xi32>` (a value) becomes a `memref<4xi32>` (a typed view of a
buffer — Chapter 3 §6 introduced memref), and ops that *produced* new
tensors become ops that *write into* buffers. After
`one-shot-bufferize` (real output):

```mlir
memref.global "private" constant @__constant_4xi32 : memref<4xi32>
    = dense<[1, 2, 3, 4]> {alignment = 64 : i64}
func.func @add_ct(%arg0: memref<4xi32, strided<[?], offset: ?>>)
    -> memref<4xi32, strided<[?], offset: ?>> {
  %0 = memref.get_global @__constant_4xi32 : memref<4xi32>
  linalg.generic ... ins(%arg0, %0 : ...) outs(%arg0 : ...) { ... }
  return %arg0 : ...
}
```

Worth a slow look:

- The constant became a **global** in read-only memory, fetched with
  `get_global` — exactly what a C compiler would do with a static array.
- `bufferizeFunctionBoundaries = true` converted the *signature* too
  (tensor arguments → memref arguments) — the same "types hide in
  signatures" issue as Chapter 10 §4, handled by an option here. The
  `strided<[?], offset: ?>` layout means "accept any stride/offset" — a
  caller-friendly ABI for function arguments.
- The `linalg.generic` now writes its result **into `%arg0`** — the
  "one-shot" analysis proved it could reuse the input buffer instead of
  allocating. Bufferization is an optimization problem (minimize copies
  and allocations), and one-shot-bufferize solves it whole-module in one
  analysis — its predecessor was a
  [fleet of per-dialect passes](https://mlir.llvm.org/docs/Passes/#bufferization-passes)
  coordinated through a dedicated
  [`bufferization` dialect](https://mlir.llvm.org/docs/Dialects/BufferizationOps/)
  of intermediate ops, which is why the name celebrates being *one* pass.
- Where buffers *are* allocated, someone must free them:
  `buildBufferDeallocationPipeline` appends the official
  [ownership-based buffer deallocation](https://mlir.llvm.org/docs/OwnershipBasedBufferDeallocation/)
  passes that insert `dealloc`s — compiler-generated
  memory management. (The helper is the official bundle of what would
  otherwise be five hand-listed passes.)

### Stages 4–5: down the familiar staircase

With buffer semantics,
[`convert-linalg-to-loops`](https://mlir.llvm.org/docs/Passes/#-convert-linalg-to-loops)
turns each `linalg.generic` into `scf.for` loops over memrefs (it
*requires* bufferized input — see section 6's war story; sibling passes
emit
[affine loops](https://mlir.llvm.org/docs/Passes/#-convert-linalg-to-affine-loops)
or
[parallel loops](https://mlir.llvm.org/docs/Passes/#-convert-linalg-to-parallel-loops)
instead). From here it's Chapter 2
§8 verbatim, extended for memory: `scf` → `cf` → `llvm`
([`convert-scf-to-cf`](https://mlir.llvm.org/docs/Passes/#-convert-scf-to-cf),
[`convert-cf-to-llvm`](https://mlir.llvm.org/docs/Passes/#-convert-cf-to-llvm));
scalar arith → `llvm`; `func` → `llvm.func`
([`convert-func-to-llvm`](https://mlir.llvm.org/docs/Passes/#-convert-func-to-llvm));
[`finalize-memref-to-llvm`](https://mlir.llvm.org/docs/Passes/#-finalize-memref-to-llvm)
lowers memref ops
to pointer arithmetic (and the `index` type to `i64`); and
[`reconcile-unrealized-casts`](https://mlir.llvm.org/docs/Passes/#-reconcile-unrealized-casts)
sweeps the temporary casts — Chapter 10 §6
taught you exactly what those are.
[`expand-strided-metadata`](https://mlir.llvm.org/docs/Passes/#-expand-strided-metadata),
the one
oddball, pre-chews fancy memref ops (like the subviews `tensor.pad`
lowering produces) into primitive ones the finalize pass can handle.

## 3. Step: run the pipeline

```bash
$TUTORIAL_OPT tests/poly_to_llvm.mlir --poly-to-llvm
```

The output is pure `llvm` dialect. The top of our small example (real
output):

```mlir
llvm.mlir.global private constant @__constant_4xi32(dense<[1, 2, 3, 4]>
    : tensor<4xi32>) {addr_space = 0 : i32, alignment = 64 : i64}
    : !llvm.array<4 x i32>
llvm.func @add_ct(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64,
                  %arg3: i64, %arg4: i64)
    -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> {
```

Look at that signature: one `memref` argument exploded into **five**
LLVM arguments (allocated pointer, aligned pointer, offset, size,
stride), and the memref return became a five-field struct. This is the
*memref calling convention*, and it's why the repo's runnable tests
export functions with **scalar** signatures (`(i32) -> i32`) — plain C
can call those without knowing MLIR's struct layout.

One more sharp edge, reproducible today: functions that *return* a
polynomial (like `add_ct`) can make the deallocation pipeline insert a
`bufferization.clone` op — and nothing later in the pipeline lowers it,
so `mlir-translate` fails with `Dialect 'bufferization' not found for
custom op 'bufferization.clone'` (verified). The pipeline is honest
teaching code: scalar-out functions work end to end;
buffer-returning ABIs are left as real-world homework.

## 4. Step: out of MLIR, into an executable

The test program,
[`tests/poly_to_llvm.mlir`](../tests/poly_to_llvm.mlir) — note it
exercises *every* `poly` op:

***tests/poly_to_llvm.mlir*** (excerpt)
```mlir
func.func @test_poly_fn(%arg : i32) -> i32 {
  %tens = tensor.splat %arg : tensor<10xi32>
  %input = poly.from_tensor %tens : tensor<10xi32> -> !poly.poly<10>
  %0 = poly.constant dense<[2, 3, 4]> : tensor<3xi32> : !poly.poly<10>
  %1 = poly.add %0, %input : !poly.poly<10>
  %2 = poly.mul %1, %1 : !poly.poly<10>
  %3 = poly.sub %2, %input : !poly.poly<10>
  %4 = poly.eval %3, %arg: (!poly.poly<10>, i32) -> i32
  return %4 : i32
}
```

and a perfectly ordinary C caller,
[`tests/poly_to_llvm_main.c`](../tests/poly_to_llvm_main.c):

***tests/poly_to_llvm_main.c***
```c
#include <stdio.h>

// This is the function we want to call from LLVM
int test_poly_fn(int x);

int main(int argc, char *argv[]) {
  int i = 1;
  int result = test_poly_fn(i);
  printf("Result: %d\n", result);
  return 0;
}
```

Four commands connect them (each verified; run from a scratch directory,
with the LLVM tools on `$PATH` per Chapter 2 §2):

```bash
# 1. MLIR -> LLVM IR (textual): leave MLIR-land
$TUTORIAL_OPT tests/poly_to_llvm.mlir --poly-to-llvm \
  | mlir-translate --mlir-to-llvmir > poly_fn.ll

# 2. LLVM IR -> native object file (--relocation-model=pic is load-bearing:
#    current toolchains, notably macOS/arm64, want PIC objects at link time)
llc --relocation-model=pic -filetype=obj < poly_fn.ll > poly_fn.o

# 3. Compile the C caller; link the two
clang -c tests/poly_to_llvm_main.c -o main.o
clang main.o poly_fn.o -o a.out

# 4. Run it
./a.out
```

```
Result: 351
```

`mlir-translate` is the border crossing: not a pass but a *translation*
(`llvm` dialect → real LLVM IR text — the official docs on
[generating LLVM IR](https://mlir.llvm.org/docs/TargetLLVMIR/) cover this
handoff); past it, MLIR is out of the
picture and the classic toolchain (`llc` to compile, `clang` to link)
takes over. Before running step 4, check the number: at x=1, the splat
polynomial p has ten 1-coefficients so p(1) = 10; c = 2+3x+4x² gives
c(1) = 9; the program computes (c+p)² − p, so (9+10)² − 10 = **351**.
The machine agrees with the algebra.

It's worth admiring what step 1's textual LLVM IR looks like for the
eval loop (from the smaller
[`tests/poly_to_llvm_eval.mlir`](../tests/poly_to_llvm_eval.mlir),
verified):

```llvm
define i32 @test_poly_fn(i32 %0) {
  br label %2
2:
  %3 = phi i64 [ %12, %6 ], [ 1, %1 ]
  %4 = phi i32 [ %11, %6 ], [ 0, %1 ]
  %5 = icmp slt i64 %3, 4
  br i1 %5, label %6, label %13
6:
  %7 = sub i64 3, %3
  %8 = mul i32 %0, %4
  %9 = getelementptr i32, ptr @__constant_3xi32, i64 %7
  %10 = load i32, ptr %9, align 4
  %11 = add i32 %8, %10
  %12 = add i64 %3, 1
  br label %2
13:
  ret i32 %4
}
```

Chapter 10's Horner `scf.for` is now branches and **phi nodes** — where
MLIR used block arguments (`^bb3(%7: i64, ...)`, Chapter 2 §8), LLVM
uses `phi`; same concept, older notation — and the coefficient tensor is
a `getelementptr` + `load` from a global. You can read your whole
compiler's work in one screen.

## 5. The lit test: five RUN lines to a running binary

Recall from Chapter 2 that lit executes each `// RUN:` line as a shell
command in a per-test sandbox, substituting placeholders like `%s` and
`%t` — so a sequence of RUN lines is a small script. The RUN header of
`poly_to_llvm.mlir` uses that to script exactly section 4 — compile,
link, run, check the printed output:

***tests/poly_to_llvm.mlir*** (excerpt)
```mlir
// RUN: tutorial-opt --poly-to-llvm %s | mlir-translate --mlir-to-llvmir | llc --relocation-model=pic -filetype=obj > %t
// RUN: clang -c %project_source_dir/tests/poly_to_llvm_main.c
// RUN: clang poly_to_llvm_main.o %t -o a.out
// RUN: ./a.out | FileCheck %s

// CHECK: 351
```

Every lit feature here is from Chapter 2 — `%t`, chained RUN lines,
FileCheck on the *runtime* output — plus one: `%project_source_dir`, the
repo-defined substitution (declared in both `lit.cfg.py` and
`lit.cmake.cfg.py`, as Chapter 2 §6 mentioned) that locates the `.c`
file from the test's sandbox. This is the fully-armed version of
Chapter 2 §8's functional testing: not an interpreter this time, but an
actual linked binary. Run both:

```bash
bazel test //tests:poly_to_llvm.mlir.test //tests:poly_to_llvm_eval.mlir.test  # Bazel
llvm-lit -sv build-ninja/tests --filter poly_to_llvm                           # CMake
```

## 6. War stories (read before composing your own pipeline)

This codebase's author documented the dead ends met while composing this
pipeline, and they're the most reusable part:

- **Pass order: `func-to-llvm` wants to go last.** Placed early, it
  converts function types before `scf`/`cf` are lowered, and the
  leftovers can't legalize. The symptom is baffling; the fix
  (section 2's stage 5 order) looks obvious only afterward.
- **"expected linalg op with buffer semantics."**
  `convert-linalg-to-loops` silently requires bufferization *before* it.
  The error names neither bufferization nor the fix — recognize it by
  symptom.
- **A plague of `unrealized_conversion_cast i64 → index`.** Cured by
  `finalize-memref-to-llvm` (which lowers `index` itself) followed by
  `reconcile-unrealized-casts` — Chapter 2's mysterious final pass, now
  fully demystified.
- **The best one: an off-by-one caught by running the program.** The
  original eval lowering computed 320 instead of 351 — a wrong loop bound
  in Horner's method. Every syntactic test passed; the *binary* didn't.
  That is the whole argument of Chapter 2 §8 (and Chapter 10's
  exercise 4), delivered by this book's own code: lowerings lie until you
  execute them.

## Where to go next

The `poly` story is complete: from custom syntax (Chapter 5) to a
number printed by a C program. The next chapter changes
*direction*: instead of lowering, it climbs up — a second dialect
(`noisy`) whose point is program **analysis**: dataflow-tracking noise
growth through arithmetic, and a genuinely *global* optimization
(placing expensive noise-reduction ops via an ILP solver) that no local
pattern could express. That's
[Chapter 12: A Global Optimization and Dataflow Analysis](12-global-optimization-and-dataflow-analysis.md)
— and the reason this repo's build pulls in Google's or-tools.

**Exercises**

1. Reproduce section 4 end to end. Then change the constant to
   `dense<[1, 2, 3]>` and *predict* the output before running — c(1)
   becomes 6, so (6+10)² − 10 = 246 (verified). If your prediction was
   wrong, so was the author's once; that's what the test is for.
2. Rerun with `--mlir-print-ir-after-all` and skim all ~20 snapshots.
   Find the exact pass where (a) the last `tensor` type disappears,
   (b) the first `llvm.` op appears, (c) the function signature changes.
3. Adapt Chapter 2 §8's approach: instead of `llc`+`clang`, feed the
   pipeline output to `mlir-runner`. You'll need a zero-argument wrapper
   function (mlir-runner can't pass arguments): write
   `@main() -> i32` calling `@test_poly_fn(1)`, and check you get 351.
4. Reproduce section 3's `bufferization.clone` failure (a function
   returning `!poly.poly<4>`), then find *two* different ways to make it
   compile: (a) change the function's ABI, (b) add an upstream pass to
   the pipeline that lowers clones (hint: look for
   `convert-bufferization-to-memref`).
5. The pipeline runs canonicalize twice and cleanup at the end. Delete
   the final cleanup quartet (conceptually) — which artifacts would
   survive in the LLVM IR? Check your answer against the actual `.ll`
   with and without `--mlir-print-ir-before=symbol-dce`.
