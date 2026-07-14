# Chapter 6: Using Traits

[Chapter 5](05-defining-a-new-dialect.md) left the `poly` dialect able to
parse and print — and do absolutely nothing else. This chapter makes the
first payments on that: by declaring **traits** on our ops, a fleet of
upstream optimization passes — CSE, loop-invariant code motion,
control-flow sinking — starts working on `poly` code *without us writing a
line of pass logic*. This is also where those bracketed lists we skipped
in Chapter 5's `PolyOps.td` get explained.

**What you will learn:**

- What traits are, and how they differ from interfaces.
- What `Pure` actually promises (two separate things!), and which upstream
  passes it unlocks — demonstrated live on `poly` ops.
- What `ElementwiseMappable` + container type constraints buy, and what
  they cost in the generated API.
- What `SameOperandsAndResultType` verifies and the syntax it enables.
- How traits appear in generated C++, and where their `verify` hooks live.
- That *missing* traits have visible consequences too (why CSE won't touch
  `poly.eval`).
- How to audit the rest of the upstream pass list: most entries need
  nothing from `poly`, one crashes the *tool* (a registration lesson),
  and one waits for Chapter 7.

**Prerequisites:** [Chapter 5](05-defining-a-new-dialect.md). Same build
requirements as before (`tutorial-opt`, via the `$TUTORIAL_OPT` variable
from Chapter 3).

---

## 1. Concepts: traits and interfaces

Generic passes have a problem: `--cse` (common subexpression elimination)
was written years before `poly.mul` existed. For it to safely merge two
identical `poly.mul` ops, it must know that `poly.mul` has no side effects
— computing it once instead of twice must be unobservable. But a generic
pass can't know that about an op it's never heard of, and the only safe
default is the conservative one: *assume the worst and touch nothing*.
That's why, out of the box, a freshly minted dialect is invisible to the
entire upstream optimization arsenal.

MLIR's fix is a declarative contract system with two closely related
mechanisms:

- An **[interface](https://mlir.llvm.org/docs/Interfaces/)** is a set of
  function signatures an op implements,
  giving passes a way to query or manipulate it without knowing its
  concrete type (think of `MemoryEffectOpInterface`, which lets any pass
  ask "what do you touch?").
- A **[trait](https://mlir.llvm.org/docs/Traits/)** is — pithily — *an
  interface with no methods*:
  a marker you "slap on" an operation asserting some property —
  "I have no memory effects", "all my operands and results have one type".
  Many traits also carry a *verification* routine, so the invariant they
  assert is checked on every parse and after every pass.

You apply traits primarily *to enable passes to do things with your ops* —
and secondarily to get free verification. Both dimensions show up below.

## 2. The traits on `poly`'s ops

Mechanically, traits occupy the third template parameter of `Op` — a
bracketed list of trait names (it defaults to empty, which is why
Chapter 5 could ignore it). Each name in the list is one contract in
section 1's sense: a property the op asserts to every pass, often with a
verification routine attached. Here is Chapter 5's binop base again,
this time reading the bracket:

***lib/Dialect/Poly/PolyOps.td*** (excerpt)
```tablegen
class Poly_BinOp<string mnemonic> : Op<Poly_Dialect, mnemonic,
    [Pure, ElementwiseMappable, SameOperandsAndResultType]> {
  let arguments = (ins PolyOrContainer:$lhs, PolyOrContainer:$rhs);
  let results = (outs PolyOrContainer:$output);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` qualified(type($output))";
  ...
}
```

Taking the three traits in turn:

### `Pure` — the workhorse

`Pure` is actually a bundle of two independent promises (both defined in
upstream's
[SideEffectInterfaces.td](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Interfaces/SideEffectInterfaces.td),
the file to read for the full effect vocabulary):

- **`NoMemoryEffect`** — executing this op neither reads nor writes any
  observable state. This is what lets CSE deduplicate it and dead-code
  elimination delete it when unused.
- **`AlwaysSpeculatable`** — this op may be executed *earlier* than
  written, or even when it wouldn't have executed at all (e.g. hoisted out
  of a conditional or a zero-iteration loop), without changing program
  behavior. Divisions are the classic counterexample: speculating a
  division can introduce a divide-by-zero that the original program
  guarded against.

The distinction is real and bites: this codebase's author first applied
only
`AlwaysSpeculatable` and was baffled that `--loop-invariant-code-motion`
did nothing — reading
[the pass source](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Transforms/Utils/LoopInvariantCodeMotionUtils.cpp)
revealed LICM requires *both* `isSpeculatable` *and* `isMemoryEffectFree`
(in today's source the gate is literally a call to a helper named
`isPure`). Hoisting an op out of a loop
moves it to different *and* possibly more/fewer executions; each half of
`Pure` covers one of those hazards.

### `ElementwiseMappable` — scalars to containers

Polynomial programs will want *vectors* of polynomials.
`ElementwiseMappable` declares that the op applied to containers means
"map me over the elements", letting one op definition serve
`!poly.poly<10>`, `tensor<2x!poly.poly<10>>`, and vectors alike. It works
together with the argument constraint:

***lib/Dialect/Poly/PolyOps.td*** (excerpt)
```tablegen
def PolyOrContainer : TypeOrValueSemanticsContainer<Polynomial, "poly-or-container">;
```

which relaxes "a polynomial" to "a polynomial, or a tensor/vector of
them". (Upstream once called this helper `TypeOrContainer`; it was
renamed to `TypeOrValueSemanticsContainer` with the same meaning.) From
`tests/poly_syntax.mlir` (Chapter 5's round-trip test):

***tests/poly_syntax.mlir*** (excerpt)
```mlir
%7 = tensor.from_elements %arg0, %arg1 : tensor<2x!poly.poly<10>>
%8 = poly.add %7, %7 : tensor<2x!poly.poly<10>>
```

There is a cost, visible in the generated code: with the wider constraint,
the generated accessors can no longer return
`::mlir::TypedValue<PolynomialType>` — `getLhs()` returns a plain
`::mlir::Value`, and C++ code that needs the poly type must `dyn_cast` for
itself. Declarative generality, paid for in static type information.

### `SameOperandsAndResultType` — one type to rule them

Verifies that both operands and the result all have exactly the same type
— for `poly.add`, that you never add a `!poly.poly<10>` to a
`!poly.poly<11>`. It also has two pleasant side effects:

- With all types provably equal, the assembly format only needs to spell
  one of them — this is why the binop syntax is the terse
  `poly.add %a, %b : !poly.poly<10>` rather than the earlier pre-trait
  `(type, type) -> type` (Chapter 5 §5).
- The trait implies **type inference** (you'll see
  `InferTypeOpInterface::Trait` in the generated code below): given
  operand types, MLIR can construct the op without being told the result
  type.

> **Note:** an earlier version of these ops used the looser
> `SameOperandsAndResultElementType` (same *element* type only —
> containers could mix with scalars), which admitted a mixed
> scalar-tensor add like
> `poly.add %tensor, %scalar : (tensor<2x!poly.poly<10>>, !poly.poly<10>) -> ...`.
> With `SameOperandsAndResultType` that is now **rejected**: feeding it
> in generic form produces
> `'poly.add' op requires the same type for all operands and results`
> (verified). Containers still work; they just must match on both sides.

`poly.eval` uses the parameterized cousin visible in `PolyOps.td` —
`AllTypesMatch<["point", "output"]>` — same idea, scoped to a named subset
of operands/results. (Its other trait, `Has32BitArguments`, is a
*custom* trait defined in this repo — `PolyTraits.h` — whose story is
verification, Chapter 8's subject. You already met its error message in
Chapter 5's exercise 3.)

## 3. Step: watch the upstream passes work

Time to collect. Three test files, three upstream passes, zero lines of
`poly`-specific pass code. (All passes here are registered by
`registerAllPasses()` in `tutorial-opt` — Chapter 3 §3.)

### Common subexpression elimination

[`tests/cse.mlir`](../tests/cse.mlir) computes the same product twice:

***tests/cse.mlir*** (excerpt)
```mlir
%2 = poly.mul %p0, %p0 : !poly.poly<10>
%3 = poly.mul %p0, %p0 : !poly.poly<10>
%4 = poly.add %2, %3 : !poly.poly<10>
```

```bash
$TUTORIAL_OPT -cse tests/cse.mlir
```

```mlir
%0 = poly.from_tensor %cst : tensor<3xi32> -> !poly.poly<10>
%1 = poly.mul %0, %0 : !poly.poly<10>
%2 = poly.add %1, %1 : !poly.poly<10>
```

One `poly.mul` remains, feeding both sides of the add. Only
`NoMemoryEffect` makes this legal — if `poly.mul` wrote to a log file,
deduplicating it would halve the log.

### Loop-invariant code motion

The upstream
[general transformation passes list](https://mlir.llvm.org/docs/Passes/#general-transformation-passes)
includes
[loop invariant code motion](https://mlir.llvm.org/docs/Passes/#-loop-invariant-code-motion),
which checks loop bodies for operations that don't need to be in the loop
and moves them out. [`tests/code_motion.mlir`](../tests/code_motion.mlir)
multiplies two loop-constant polynomials *inside* a loop:

***tests/code_motion.mlir*** (excerpt)
```mlir
%ret_val = affine.for %i = 0 to 100 iter_args(%sum_iter = %p0) -> !poly.poly<10> {
  %2 = poly.mul %p0, %p1 : !poly.poly<10>
  %sum_next = poly.add %sum_iter, %2 : !poly.poly<10>
  affine.yield %sum_next : !poly.poly<10>
}
```

```bash
$TUTORIAL_OPT --loop-invariant-code-motion tests/code_motion.mlir
```

```mlir
%0 = poly.from_tensor %cst : tensor<3xi32> -> !poly.poly<10>
%1 = poly.from_tensor %cst_0 : tensor<3xi32> -> !poly.poly<10>
%2 = poly.mul %0, %1 : !poly.poly<10>
%3 = affine.for %arg0 = 0 to 100 iter_args(%arg1 = %0) -> (!poly.poly<10>) {
  %4 = poly.add %arg1, %2 : !poly.poly<10>
  ...
```

The `poly.mul` is hoisted above the loop — computed 1× instead of 100×.
This is the pass that needs the *whole* of `Pure` (section 2's war
story). Note the `poly.add` correctly stays inside: it consumes
`%sum_iter`, which changes every iteration — LICM checks invariance of
operands, traits or no traits.

### Control-flow sinking

The dual of hoisting. Where LICM rescues work from a region that runs
*many* times,
[control-flow sink](https://mlir.llvm.org/docs/Passes/#-control-flow-sink)
pushes work into a region that may run *zero* times: an operation whose
only uses sit inside one conditionally-executed region is moved into
that region, so it executes only when its result is actually consumed.
[`tests/control_flow_sink.mlir`](../tests/control_flow_sink.mlir) builds
two polynomials *before* a branch, but each is used in only one arm of
an `scf.if` (Chapter 1 §4's structured conditional):

***tests/control_flow_sink.mlir***
```mlir
func.func @test_simple_sink(%arg0: i1) -> !poly.poly<10> {
  %0 = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
  %p0 = poly.from_tensor %0 : tensor<3xi32> -> !poly.poly<10>
  %1 = arith.constant dense<[9, 8, 16]> : tensor<3xi32>
  %p1 = poly.from_tensor %1 : tensor<3xi32> -> !poly.poly<10>
  %4 = scf.if %arg0 -> (!poly.poly<10>) {
    %2 = poly.mul %p0, %p0 : !poly.poly<10>
    scf.yield %2 : !poly.poly<10>
  } else {
    %3 = poly.mul %p1, %p1 : !poly.poly<10>
    scf.yield %3 : !poly.poly<10>
  }
  return %4 : !poly.poly<10>
}
```

```bash
$TUTORIAL_OPT -control-flow-sink tests/control_flow_sink.mlir
```

```mlir
func.func @test_simple_sink(%arg0: i1) -> !poly.poly<10> {
  %0 = scf.if %arg0 -> (!poly.poly<10>) {
    %cst = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
    %1 = poly.from_tensor %cst : tensor<3xi32> -> !poly.poly<10>
    %2 = poly.mul %1, %1 : !poly.poly<10>
    scf.yield %2 : !poly.poly<10>
  } else {
    %cst = arith.constant dense<[9, 8, 16]> : tensor<3xi32>
    %1 = poly.from_tensor %cst : tensor<3xi32> -> !poly.poly<10>
    %2 = poly.mul %1, %1 : !poly.poly<10>
    scf.yield %2 : !poly.poly<10>
  }
  return %0 : !poly.poly<10>
}
```

Comparing before and after:

- **Nothing is left in front of the branch.** Each `poly.from_tensor`
  moved into the arm that uses it, so whichever way `%arg0` goes at
  runtime, the untaken arm's polynomial is never materialized — the
  work runs "only if used" instead of "always".
- **The `arith.constant`s went along for the ride.** Sinking chases
  whole def chains: once a `from_tensor` sinks, its constant's only use
  sits inside that arm as well, making the constant sinkable too. (It
  qualifies for the same reason — `arith.constant` carries `Pure`
  upstream.)
- **The licensing trait is `Pure` again, seen from the other side.**
  LICM moved an op to *more* executions — that's why it also demands
  the speculatability half (§2's war story). Sinking moves an op to
  *fewer*, and the half doing the work is `NoMemoryEffect`: skipping an
  execution is only invisible when the op has no effects to skip. If
  `poly.from_tensor` appended to a log file, sinking it would silence
  the log every time the other arm ran.

One thing sinking will *not* do is duplicate work. Rewrite the test so
*both* arms use the same polynomial — `%p0` squared in one arm, added to
itself in the other:

```mlir
%p0 = poly.from_tensor %0 : tensor<3xi32> -> !poly.poly<10>
%4 = scf.if %arg0 -> (!poly.poly<10>) {
  %2 = poly.mul %p0, %p0 : !poly.poly<10>
  scf.yield %2 : !poly.poly<10>
} else {
  %3 = poly.add %p0, %p0 : !poly.poly<10>
  scf.yield %3 : !poly.poly<10>
}
```

and the pass leaves the IR exactly as it found it (verified). The pass
only *moves* ops, never clones them, so an op can sink only when **all**
of its uses land in a single region — `%p0`'s uses span both arms, so it
stays where it is, traits or no traits. Same lesson as LICM declining to
hoist the `poly.add`: traits grant *permission*; the pass's own
profitability and correctness analysis still decides.

Run all three as tests:

```bash
bazel test //tests:cse.mlir.test //tests:code_motion.mlir.test \
           //tests:control_flow_sink.mlir.test          # Bazel
llvm-lit -sv build-ninja/tests --filter 'cse|code_motion|control_flow_sink'  # CMake
```

```
Total Discovered Tests: 18
  Excluded: 15 (83.33%)
  Passed  :  3 (16.67%)
```

### The dog that didn't bark

Traits you *don't* declare also teach. `poly.eval`'s trait list has no
`Pure`. Duplicate an eval and run CSE:

```mlir
%0 = poly.eval %p, %x : (!poly.poly<10>, i32) -> i32
%1 = poly.eval %p, %x : (!poly.poly<10>, i32) -> i32
%2 = arith.addi %0, %1 : i32
```

Both `poly.eval` ops survive `-cse` (verified). Nothing is wrong with
`eval` — it's a pure function mathematically — but nobody *told* MLIR
that, so the conservative default rules. When your shiny new op
mysteriously escapes optimization, the first place to look is its trait
list.

### The rest of the pass list

The three demos weren't cherry-picked — they're what remains after
reading the
[general transformation passes list](https://mlir.llvm.org/docs/Passes/#general-transformation-passes)
end to end and asking, for each entry, *what does this pass need from
`poly`?* The others sort into three buckets, and the sorting itself is
a skill worth having whenever you bring up a new dialect:

- **Nothing — the pass works on structures `poly` doesn't have.**
  [`-mem2reg`](https://mlir.llvm.org/docs/Passes/#-mem2reg) (promotes
  memory slots into SSA values) and
  [`-sroa`](https://mlir.llvm.org/docs/Passes/#-sroa) (scalar
  replacement of aggregates) rewrite memory operations, and `poly`
  programs contain none;
  [`-remove-dead-values`](https://mlir.llvm.org/docs/Passes/#-remove-dead-values)
  trims unused function arguments and results;
  [`-symbol-dce`](https://mlir.llvm.org/docs/Passes/#-symbol-dce)
  deletes private functions nobody references. Stack all four onto a
  `poly` file and nothing changes (verified):

  ```bash
  $TUTORIAL_OPT -mem2reg -sroa -remove-dead-values -symbol-dce tests/cse.mlir
  ```

  They aren't *failing* on `poly` — they have nothing to do here, and
  no trait would change that.

- **Something from the *tool*, not the dialect.**
  [`-inline`](https://mlir.llvm.org/docs/Passes/#-inline) also has
  nothing `poly`-specific to do — it rearranges function calls, and
  `poly` ops would just ride along — but try it and this repo's binary
  *crashes*:

  ```bash
  $TUTORIAL_OPT -inline tests/cse.mlir
  ```

  ```
  LLVM ERROR: checking for an interface (`mlir::DialectInlinerInterface`)
  that was promised by dialect 'func' but never implemented. This is
  generally an indication that the dialect extension implementing the
  interface was never registered.
  ```

  Decode it with section 1's vocabulary: inlining is powered by an
  *interface* — attached to a whole dialect rather than a single op —
  that answers questions like "may this body be inlined into that call
  site?". The `func` dialect *promises* that interface but ships the
  implementation in a separate extension, which upstream `mlir-opt`
  registers and `tutorial-opt` never does — the same file inlines fine
  under stock `mlir-opt` (verified with a two-function example). A
  useful failure to have met: "the pass doesn't work" sometimes means a
  missing trait, and sometimes a missing *registration* — Chapter 5
  §3's lesson resurfacing one level up.

- **Folding — the next chapter's subject.**
  [`-sccp`](https://mlir.llvm.org/docs/Passes/#-sccp) (sparse
  conditional constant propagation) propagates *values*: to push
  constants through `poly.mul` it must ask the op to compute its
  result, and nothing taught so far provides that. The machinery is
  called a *folder*, and it's Chapter 7. (If you can't wait: the repo's
  ops already carry Chapter 7's folders, so
  `$TUTORIAL_OPT -sccp tests/sccp.mlir` already rewrites the constant
  `poly` arithmetic into `poly.constant`s today — verified.)

## 4. Under the hood

What does the bracket in tablegen become? Every trait is a C++ *template
mixin* threaded into the op's base class — the same CRTP dance as
Chapters 3–4, at industrial scale. To see it, run Chapter 5's
`--gen-op-decls` command again and look at `AddOp`'s actual base class
(verified, one line, wrapped here):

```cpp
class AddOp : public ::mlir::Op<AddOp,
    ::mlir::OpTrait::ZeroRegions,
    ::mlir::OpTrait::OneResult,
    ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl,
    ::mlir::OpTrait::ZeroSuccessors,
    ::mlir::OpTrait::NOperands<2>::Impl,
    ::mlir::OpTrait::OpInvariants,
    ::mlir::ConditionallySpeculatable::Trait,
    ::mlir::OpTrait::AlwaysSpeculatableImplTrait,
    ::mlir::MemoryEffectOpInterface::Trait,
    ::mlir::OpTrait::Elementwise,
    ::mlir::OpTrait::Scalarizable,
    ::mlir::OpTrait::Vectorizable,
    ::mlir::OpTrait::Tensorizable,
    ::mlir::OpTrait::SameOperandsAndResultType,
    ::mlir::InferTypeOpInterface::Trait> {
```

You can see the expansions, declaration by declaration:

- `Pure` became the
  `ConditionallySpeculatable`/`AlwaysSpeculatableImplTrait` pair plus
  `MemoryEffectOpInterface` (whose generated `getEffects` body is empty —
  "no effects" as literal code).
- `ElementwiseMappable` became
  `Elementwise` + `Scalarizable` + `Vectorizable` + `Tensorizable`.
- `SameOperandsAndResultType` dragged in `InferTypeOpInterface` as
  promised.
- Even "trivia" you never declared is trait-encoded: `ZeroRegions`,
  `OneResult`, `NOperands<2>` come from the `arguments`/`results` you
  wrote in Chapter 5.

Mechanically, a trait is a class template with optional hooks; the most
common is `verifyTrait`, which runs as part of op verification. That hook
is how `SameOperandsAndResultType` rejects mismatched types — and it's the
hook our own `Has32BitArguments` implements, as Chapter 8 will show in
detail.

A practical note, still true: there is no single
documented list of all upstream traits. The Traits documentation's
[operation traits list](https://mlir.llvm.org/docs/Traits/#operation-traits-list)
covers many,
but some (e.g. `ConstantLike`, `Involution`, `Idempotent`) you discover
only by reading `OpBase.td` and the pass sources. A few worth knowing
exist, even though `poly` doesn't use them: `Commutative` (operand
reordering; canonicalization uses it to move constants rightward — the
fact `PowerOfTwoExpand` relied on in Chapter 3!), `Involution`
(`f(f(x)) = x`, would auto-cancel a hypothetical double `poly.neg`), and
`Idempotent` (`f(f(x)) = f(x)`).

Skimming the list also turns up whole families for capabilities `poly`'s
ops don't have — regions and symbols — though you have been *consuming*
them since Chapter 1: `IsolatedFromAbove` (Chapter 3 §4's isolation
contract, the reason `func.func` can anchor a parallel pass),
`Terminator` plus `SingleBlock`/`SingleBlockImplicitTerminator` (the
rules that every block ends in a `return`/`scf.yield`/`affine.yield` —
and that a trivial terminator may be inserted for you), `AffineScope`
(also on `func.func`: the boundary inside which `affine`'s
symbol-and-dimension rules apply), `SymbolTable` (on `builtin.module`;
what makes lookups of `@main`-style names work), and `Broadcastable`
(mixed-shape arithmetic for ML-style tensor ops — `poly`'s containers,
by contrast, must match exactly, as `SameOperandsAndResultType`
insists). None of these belong on `poly.add`; recognizing their names in
other dialects' tablegen is the point.

## Where to go next

Traits let *existing* ops be moved, merged, and deleted — but nothing yet
*computes* with polynomials at compile time. `tests/sccp.mlir` is sitting
in the repo waiting: sparse conditional constant propagation can replace
`poly.mul` of known constants with a `poly.constant` of the product — once
the ops know how to **fold**. That, plus the `hasConstantMaterializer`
flag we skipped in Chapter 5's dialect shell, is
[Chapter 7: Folders and Constant Propagation](07-folders-and-constant-propagation.md).

**Exercises**

1. Reproduce the "dog that didn't bark": write the duplicate-`poly.eval`
   function from section 3, run `-cse`, and confirm both evals survive.
   Then find, in `PolyOps.td`, exactly what you would change so CSE could
   merge them.
2. Make LICM refuse: change `tests/code_motion.mlir`'s `poly.mul` to
   depend on `%sum_iter` and confirm the mul stays in the loop — trait or
   not, invariance is about operands.
3. Verify the stricter trait yourself: run this chapter's mixed
   scalar/tensor `poly.add` (§2's note; use the generic `"poly.add"(...)`
   syntax from
   Chapter 1 §3, since the pretty syntax can't even express it) and read
   the verifier error.
4. Run `-cse` and `--loop-invariant-code-motion` *together* on
   `tests/cse.mlir` and `tests/code_motion.mlir` with two `--pass-pipeline`
   entries or sequential flags — does order matter for these two files?
   Predict, then check.
5. Skim [mlir.llvm.org/docs/Traits](https://mlir.llvm.org/docs/Traits/)
   and pick one trait not used by `poly` (e.g. `Involution`). Sketch the
   `.td` for a `poly.neg` op that would exploit it, and what
   canonicalization it would enable for `poly.neg (poly.neg %x)`.
