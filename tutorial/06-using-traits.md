# Tutorial 6: Using Traits

This is a step-by-step companion to the article
[Using Traits](https://jeremykun.com/2023/09/07/mlir-using-traits/).
[Tutorial 5](05-defining-a-new-dialect.md) left the `poly` dialect able to
parse and print — and do absolutely nothing else. This tutorial makes the
first payments on that: by declaring **traits** on our ops, a fleet of
upstream optimization passes — CSE, loop-invariant code motion,
control-flow sinking — starts working on `poly` code *without us writing a
line of pass logic*. This is also where those bracketed lists we skipped
in Tutorial 5's `PolyOps.td` get explained.

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

**Prerequisites:** [Tutorial 5](05-defining-a-new-dialect.md). Same build
requirements as before (`tutorial-opt`, via the `$TUTORIAL_OPT` variable
from Tutorial 3).

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

- An **interface** is a set of function signatures an op implements,
  giving passes a way to query or manipulate it without knowing its
  concrete type (think of `MemoryEffectOpInterface`, which lets any pass
  ask "what do you touch?").
- A **trait** is, in the article's words, *an interface with no methods*:
  a marker you "slap on" an operation asserting some property —
  "I have no memory effects", "all my operands and results have one type".
  Many traits also carry a *verification* routine, so the invariant they
  assert is checked on every parse and after every pass.

You apply traits primarily *to enable passes to do things with your ops* —
and secondarily to get free verification. Both dimensions show up below.

## 2. The traits on `poly`'s ops

Mechanically, traits occupy the third template parameter of `Op` — a
bracketed list of trait names (it defaults to empty, which is why
Tutorial 5 could ignore it). Each name in the list is one contract in
section 1's sense: a property the op asserts to every pass, often with a
verification routine attached. Here is Tutorial 5's binop base again,
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

`Pure` is actually a bundle of two independent promises:

- **`NoMemoryEffect`** — executing this op neither reads nor writes any
  observable state. This is what lets CSE deduplicate it and dead-code
  elimination delete it when unused.
- **`AlwaysSpeculatable`** — this op may be executed *earlier* than
  written, or even when it wouldn't have executed at all (e.g. hoisted out
  of a conditional or a zero-iteration loop), without changing program
  behavior. Divisions are the classic counterexample: speculating a
  division can introduce a divide-by-zero that the original program
  guarded against.

The distinction is real and bites: the article's author first applied only
`AlwaysSpeculatable` and was baffled that `--loop-invariant-code-motion`
did nothing — reading the pass source revealed LICM requires *both*
`isSpeculatable` *and* `isMemoryEffectFree`. Hoisting an op out of a loop
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
them". From `tests/poly_syntax.mlir` (Tutorial 5's round-trip test):

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
  `poly.add %a, %b : !poly.poly<10>` rather than Tutorial 5's article-era
  `(type, type) -> type`.
- The trait implies **type inference** (you'll see
  `InferTypeOpInterface::Trait` in the generated code below): given
  operand types, MLIR can construct the op without being told the result
  type.

`poly.eval` uses the parameterized cousin visible in `PolyOps.td` —
`AllTypesMatch<["point", "output"]>` — same idea, scoped to a named subset
of operands/results. (Its other trait, `Has32BitArguments`, is a
*custom* trait defined in this repo — `PolyTraits.h` — whose story is
verification, Tutorial 8's subject. You already met its error message in
Tutorial 5's exercise 3.)

## 3. Step: watch the upstream passes work

Time to collect. Three test files, three upstream passes, zero lines of
`poly`-specific pass code. (All passes here are registered by
`registerAllPasses()` in `tutorial-opt` — Tutorial 3 §3.)

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

[`tests/code_motion.mlir`](../tests/code_motion.mlir) multiplies two
loop-constant polynomials *inside* a loop:

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

The dual of hoisting: [`tests/control_flow_sink.mlir`](../tests/control_flow_sink.mlir)
builds two polynomials *before* a branch, but each is used in only one arm
of an `scf.if`. Sinking moves each `poly.from_tensor` *into* the branch
that uses it:

```bash
$TUTORIAL_OPT -control-flow-sink tests/control_flow_sink.mlir
```

so the untaken branch's polynomial is never materialized. Moving an op to
*fewer* executions is speculation's mirror image — again licensed by
`Pure`.

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

## 4. Under the hood

What does the bracket in tablegen become? Every trait is a C++ *template
mixin* threaded into the op's base class — the same CRTP dance as
Tutorials 3–4, at industrial scale. To see it, run Tutorial 5's
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
  wrote in Tutorial 5.

Mechanically, a trait is a class template with optional hooks; the most
common is `verifyTrait`, which runs as part of op verification. That hook
is how `SameOperandsAndResultType` rejects mismatched types — and it's the
hook our own `Has32BitArguments` implements, as Tutorial 8 will show in
detail.

A practical note from the article, still true: there is no single
documented list of all upstream traits. The
[Traits documentation](https://mlir.llvm.org/docs/Traits/) covers many,
but some (e.g. `ConstantLike`, `Involution`, `Idempotent`) you discover
only by reading `OpBase.td` and the pass sources. A few worth knowing
exist, even though `poly` doesn't use them: `Commutative` (operand
reordering; canonicalization uses it to move constants rightward — the
fact `PowerOfTwoExpand` relied on in Tutorial 3!), `Involution`
(`f(f(x)) = x`, would auto-cancel a hypothetical double `poly.neg`), and
`Idempotent` (`f(f(x)) = f(x)`).

## Differences from the original article

- **The type-equality trait got stricter.** The article applies
  `SameOperandsAndResultElementType` (same *element* type, containers may
  mix with scalars); the repo now uses `SameOperandsAndResultType`. The
  article's mixed scalar-tensor example —
  `poly.add %tensor, %scalar : (tensor<2x!poly.poly<10>>, !poly.poly<10>) -> ...`
  — is therefore now **rejected**: feeding it in generic form produces
  `'poly.add' op requires the same type for all operands and results`
  (verified). Containers still work; they just must match on both sides.
- The article defers SCCP ("requires a bit of extra work... next time");
  the repo already contains `tests/sccp.mlir` and everything it needs —
  that extra work is folding, i.e. exactly the next tutorial.
- `PolyOrContainer` is spelled with `TypeOrValueSemanticsContainer` in the
  current repo (the article used `TypeOrContainer`) — an upstream renaming
  with the same meaning.

## Where to go next

Traits let *existing* ops be moved, merged, and deleted — but nothing yet
*computes* with polynomials at compile time. `tests/sccp.mlir` is sitting
in the repo waiting: sparse conditional constant propagation can replace
`poly.mul` of known constants with a `poly.constant` of the product — once
the ops know how to **fold**. That, plus the `hasConstantMaterializer`
flag we skipped in Tutorial 5's dialect shell, is
[Tutorial 7: Folders and Constant Propagation](07-folders-and-constant-propagation.md).

**Exercises**

1. Reproduce the "dog that didn't bark": write the duplicate-`poly.eval`
   function from section 3, run `-cse`, and confirm both evals survive.
   Then find, in `PolyOps.td`, exactly what you would change so CSE could
   merge them.
2. Make LICM refuse: change `tests/code_motion.mlir`'s `poly.mul` to
   depend on `%sum_iter` and confirm the mul stays in the loop — trait or
   not, invariance is about operands.
3. Verify the stricter trait yourself: run this tutorial's mixed
   scalar/tensor `poly.add` (use the generic `"poly.add"(...)` syntax from
   Tutorial 1 §3, since the pretty syntax can't even express it) and read
   the verifier error.
4. Run `-cse` and `--loop-invariant-code-motion` *together* on
   `tests/cse.mlir` and `tests/code_motion.mlir` with two `--pass-pipeline`
   entries or sequential flags — does order matter for these two files?
   Predict, then check.
5. Skim [mlir.llvm.org/docs/Traits](https://mlir.llvm.org/docs/Traits/)
   and pick one trait not used by `poly` (e.g. `Involution`). Sketch the
   `.td` for a `poly.neg` op that would exploit it, and what
   canonicalization it would enable for `poly.neg (poly.neg %x)`.
