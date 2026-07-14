# Chapter 7: Folders and Constant Propagation

[Chapter 6](06-using-traits.md) ended on a cliffhanger: `tests/sccp.mlir`
wants to replace `poly.mul` of *known* polynomials with the precomputed
product, and traits alone can't do that — someone has to teach MLIR to
actually *multiply polynomials at compile time*. That someone is the
**folder**, the `fold` method behind Chapter 5's `let hasFolder = 1`.

One thing to get straight up front, because the names blur together:
folding is **not a pass**, and it doesn't compete with `--canonicalize`
or `--sccp` — it's the op-level mechanism *underneath* both. You write
one `fold` method per op, saying what that op evaluates to when its
inputs are known; upstream passes then *drive* your folders with
different strategies — `--canonicalize` greedily, for local cleanup;
`--sccp` from a global dataflow analysis that pushes known constants
through branches and loops. The passes supply the strategy, the folder
supplies the arithmetic; section 1 makes that division precise.
Completing the machinery are the last two IOUs from Chapter 5: the
`poly.constant` op (a computed constant has to live *somewhere* in the
IR) and the dialect's `hasConstantMaterializer` (what lets a pass create
that op on your behalf).

**What you will learn:**

- What folding is, and its precise, deliberately limited contract.
- The three consumers of folders: the greedy rewrite driver (Chapter 3's
  `applyPatternsAndFoldGreedily`), `--canonicalize`, and `--sccp` — and how
  the last two differ.
- Why constant propagation needs a *constant op* (`ConstantLike`,
  attribute arguments) and a *constant materializer*.
- How to implement folders in C++: `FoldAdaptor`, `OpFoldResult`,
  returning `nullptr`, `APInt`, and `DenseIntElementsAttr`.
- The cyclic polynomial multiplication fold, verified down to its
  wraparound arithmetic.

**Prerequisites:** [Chapter 6](06-using-traits.md). Same build setup
(`$TUTORIAL_OPT`).

---

## 1. Concepts: folding and its consumers

**Folding** is computing an operation's result at compile time and
replacing the op with that result. Its contract is deliberately narrow —
a fold may:

- inspect only the *single* op being folded (plus constant values of its
  operands),
- return either an **attribute** (a compile-time constant result) or an
  **existing SSA value**,
- and may **not** create new ops, delete other ops, or look around the
  program.

Compare Chapter 3's rewrite patterns, which may do all of those things.
The restriction is the point: because folds are so local and safe, MLIR
can invoke them *constantly* — after creating ops in a rewrite, inside the
greedy driver (that's the "AndFold" in `applyPatternsAndFoldGreedily`, and
the reason Chapter 3's exercise 2 saw `1*x` vanish before any pattern
ran), and inside dedicated passes:

- **`--canonicalize`** runs folds (plus canonicalization patterns,
  Chapter 9's subject) greedily and deletes dead code. It is *local*: it
  cannot reason across control-flow boundaries.
- [**`--sccp`**](https://mlir.llvm.org/docs/Passes/#-sccp) —
  [*sparse conditional constant propagation*](https://en.wikipedia.org/wiki/Sparse_conditional_constant_propagation),
  a classic SSA-based optimization long predating MLIR — is the global
  consumer: it runs a dataflow analysis that tracks "is this value known
  to be a constant?" through branches and loops (it can even conclude that
  a branch is never taken and propagate through the surviving side), and
  it uses folders to compute what each op does to known-constant inputs.

A warm-up on `arith`, from the first function of
[`tests/sccp.mlir`](../tests/sccp.mlir):

***tests/sccp.mlir***
```mlir
func.func @test_arith_sccp() -> i32 {
  %0 = arith.constant 7 : i32
  %1 = arith.constant 8 : i32
  %2 = arith.addi %0, %0 : i32
  %3 = arith.muli %0, %0 : i32
  %4 = arith.addi %2, %3 : i32
  return %2 : i32
}
```

```bash
$TUTORIAL_OPT -pass-pipeline="builtin.module(func.func(sccp))" tests/sccp.mlir
```

```mlir
func.func @test_arith_sccp() -> i32 {
  %c63_i32 = arith.constant 63 : i32
  %c49_i32 = arith.constant 49 : i32
  %c14_i32 = arith.constant 14 : i32
  %c8_i32 = arith.constant 8 : i32
  %c7_i32 = arith.constant 7 : i32
  return %c14_i32 : i32
}
```

Every computable value was computed (7+7=14, 7·7=49, 14+49=63) and each op
replaced by a constant — but note **sccp does not delete dead code**; the
unused constants just sit there (a later `--cse`/`--canonicalize` sweeps
them). By contrast, `--canonicalize` on the same function leaves exactly
two lines (verified):

```mlir
func.func @test_arith_sccp() -> i32 {
  %c14_i32 = arith.constant 14 : i32
  return %c14_i32 : i32
}
```

Same folders underneath — different drivers with different scopes. Run
the second function of `sccp.mlir` through either pass today and *nothing*
happens to the `poly` ops. Three ingredients are missing.

## 2. Ingredient 1: a constant op

When sccp discovers `%2` is "the polynomial with coefficients
`[1, 4, 10, 12, 9]`", it must put that knowledge back into the IR as an
op. `arith` has `arith.constant` for this; `poly` needs its own — this is
`poly.constant` from `PolyOps.td`, whose `let`s we skipped in Chapter 5:

***lib/Dialect/Poly/PolyOps.td***
```tablegen
def Poly_ConstantOp : Op<Poly_Dialect, "constant", [Pure, ConstantLike]> {
  let summary = "Define a constant polynomial via an attribute.";
  let arguments = (ins AnyIntElementsAttr:$coefficients);
  let results = (outs Polynomial:$output);
  let assemblyFormat = "$coefficients attr-dict `:` qualified(type($output))";
  let hasFolder = 1;
}
```

Two new things:

- **An attribute argument.** Unlike every previous op, the "input" here is
  not an SSA value — `ins AnyIntElementsAttr:$coefficients` declares a
  compile-time *attribute* operand (Chapter 1 §3's generic form showed
  attributes as the `<{...}>` dictionary; Chapter 3 built one with
  `rewriter.getIntegerAttr`). `AnyIntElementsAttr` is an attribute
  *constraint*: any dense-elements attribute with integer elements, of any
  bit width. That buys the flexible syntax seen in `poly_syntax.mlir`:

  ***tests/poly_syntax.mlir*** (excerpt)
  ```mlir
  %10 = poly.constant dense<[2, 3, 4]> : tensor<3xi32> : !poly.poly<10>
  %11 = poly.constant dense<[2, 3, 4]> : tensor<3xi8>  : !poly.poly<10>
  %13 = poly.constant dense<4> : tensor<100xi32>       : !poly.poly<10>
  ```

  (A small `assemblyFormat` detail worth noticing here:
  `qualified(type($output))` makes the result type print in full —
  `!poly.poly<10>` — where a bare `type($output)` would print only the
  parameter part, `<10>`.)

- **`ConstantLike`** — a trait marking this op as "a constant" for the
  folding machinery, so the engines know its value without special-casing
  (and so it can be uniquely rebuilt from an attribute + type). The
  benefits of having a dedicated constant op are explained in the
  [MLIR documentation on folding](https://mlir.llvm.org/docs/Canonicalization/#canonicalizing-with-the-fold-method).

## 3. Ingredient 2: fold methods

`let hasFolder = 1` (on `constant`, `from_tensor`, `add`, `sub`, `mul`)
makes tablegen emit a declaration you must implement:

```cpp
OpFoldResult MulOp::fold(MulOp::FoldAdaptor adaptor);
```

(The signature would be different if the op had more than one result value
— see [the folding docs](https://mlir.llvm.org/docs/Canonicalization/#canonicalizing-with-the-fold-method).)

The **`FoldAdaptor`** is a shim with the same accessor names as the op —
`getLhs()`, `getOperands()` — except each returns an `Attribute`: the
constant value of that operand *if the engine knows it*, or null if it
doesn't. Your job: if enough operands are known, compute the result and
return it; otherwise return `nullptr`, meaning "this fold does not apply"
(the op stays untouched). The implementations live in
[`lib/Dialect/Poly/PolyOps.cpp`](../lib/Dialect/Poly/PolyOps.cpp), in
increasing order of interest. (That file also contains `EvalOp::verify`
and `getCanonicalizationPatterns` bodies — Chapters 8 and 9; ignore them
today.)

***lib/Dialect/Poly/PolyOps.cpp***
```cpp
OpFoldResult ConstantOp::fold(ConstantOp::FoldAdaptor adaptor) {
  return adaptor.getCoefficients();
}
```

A constant folds to its own attribute — this one-liner is what plugs
`poly.constant` into the engine's "what value is this?" query.

***lib/Dialect/Poly/PolyOps.cpp***
```cpp
OpFoldResult FromTensorOp::fold(FromTensorOp::FoldAdaptor adaptor) {
  // Returns null if the cast failed, which corresponds to a failed fold.
  return dyn_cast_or_null<DenseIntElementsAttr>(adaptor.getInput());
}
```

If the input tensor is a known dense-integer constant, the polynomial *is*
that attribute; if the input isn't constant (`getInput()` returns null) or
is some other attribute kind, `dyn_cast_or_null` yields `nullptr` — a
declined fold, in one expression.

***lib/Dialect/Poly/PolyOps.cpp***
```cpp
OpFoldResult AddOp::fold(AddOp::FoldAdaptor adaptor) {
  return constFoldBinaryOp<IntegerAttr, APInt, void>(
      adaptor.getOperands(), [&](APInt a, APInt b) { return a + b; });
}
```

Addition of polynomials is elementwise on coefficients, and upstream
provides `constFoldBinaryOp` (from `mlir/Dialect/CommonFolders.h`) to do
the elementwise plumbing: hand it the operand attributes and a lambda on
**`APInt`** — LLVM's arbitrary-precision integer, which carries its bit
width and wraps accordingly (our "coefficients mod 2³²" semantics fall out
of 32-bit `APInt` arithmetic for free). `sub` is the same with `a - b`.
(The third template parameter, `void`, is a newer-LLVM addition — older
examples of this helper show only two.)

Multiplication is the real one — naive textbook polynomial multiplication,
in the ring ℤ[x]/(xᴺ − 1):

***lib/Dialect/Poly/PolyOps.cpp***
```cpp
OpFoldResult MulOp::fold(MulOp::FoldAdaptor adaptor) {
  auto lhs = dyn_cast_or_null<DenseIntElementsAttr>(adaptor.getOperands()[0]);
  auto rhs = dyn_cast_or_null<DenseIntElementsAttr>(adaptor.getOperands()[1]);

  if (!lhs || !rhs) return nullptr;

  auto degree = llvm::cast<PolynomialType>(getResult().getType()).getDegreeBound();
  auto maxIndex = lhs.size() + rhs.size() - 1;

  SmallVector<APInt, 8> result;
  result.reserve(maxIndex);
  for (int i = 0; i < maxIndex; ++i) {
    result.push_back(APInt((*lhs.begin()).getBitWidth(), 0));
  }

  int i = 0;
  for (auto lhsIt = lhs.value_begin<APInt>(); lhsIt != lhs.value_end<APInt>();
       ++lhsIt) {
    int j = 0;
    for (auto rhsIt = rhs.value_begin<APInt>(); rhsIt != rhs.value_end<APInt>();
         ++rhsIt) {
      // index is modulo degree because poly's semantics are defined modulo x^N
      // = 1.
      result[(i + j) % degree] += *rhsIt * (*lhsIt);
      ++j;
    }
    ++i;
  }

  return DenseIntElementsAttr::get(
      RankedTensorType::get(static_cast<int64_t>(result.size()),
                            IntegerType::get(getContext(), 32)),
      result);
}
```

Read it in execution order. The null guards up front decline the fold
when either side isn't a known constant (we'll watch that happen in
section 5) — battle scars: an earlier version of this code `cast<>`ed the
operands directly and crashed whenever one wasn't constant. (Note also the
free-function `llvm::cast<PolynomialType>(...)` spelling: the
member-function form `type.cast<PolynomialType>()` was deprecated
upstream.) Past the guards, the code prepares a result vector of
`lhs.size() + rhs.size() - 1` coefficients, each zero-initialized as
`APInt(bitwidth, 0)` — the bit width must match the operands', because
`APInt`s of different widths don't mix. The double loop is convolution:
`DenseIntElementsAttr` values are iterated as `APInt` via
`value_begin<APInt>()`, coefficient `i` times coefficient `j` lands at
index `i+j`, and the `% degree` implements the ring's wraparound —
xᴺ ≡ 1, so x¹² in a degree-10 ring is x². Finally the vector is packaged
as the returned attribute, and the result attribute needs a *type* —
attributes are typed, so the code conjures a `RankedTensorType` of the
right size. (Note it builds one of length `lhs.size() + rhs.size() - 1`
even though wraparound means anything past `degree` is zero — a small
infelicity we'll actually observe in section 5.)

## 4. Ingredient 3: the constant materializer

A fold produced the attribute `dense<[1, 4, 10, 12, 9]>` — but an
attribute is not an op. Something must turn it back into IR, and folds
themselves are forbidden from creating ops. That is the dialect-level
**constant materializer** — `let hasConstantMaterializer = 1` in
`PolyDialect.td` (the last unexplained line from Chapter 5!), implemented
in [`PolyDialect.cpp`](../lib/Dialect/Poly/PolyDialect.cpp):

***lib/Dialect/Poly/PolyDialect.cpp***
```cpp
Operation *PolyDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                            Type type, Location loc) {
  auto coeffs = dyn_cast<DenseIntElementsAttr>(value);
  if (!coeffs)
    return nullptr;
  return builder.create<ConstantOp>(loc, type, coeffs);
}
```

The engine hands the dialect an attribute and the type the value must
have, and the dialect decides which op represents it: the `dyn_cast`
guard returns `nullptr` for any attribute kind that isn't the
dense-integer form our folds produce — declining, just as a fold does —
and otherwise builds a `poly.constant`. (The `Type` argument matters
because the same attribute can produce multiple different types, e.g. via
different interpretations of a hex string or
[splatting](https://mlir.llvm.org/doxygen/classmlir_1_1SplatElementsAttr.html)
into result tensors of different dimensions.) The division of labor is now
complete:

> folders **compute** attributes → the materializer turns attributes into
> **constant ops** → `ConstantLike`/`ConstantOp::fold` let engines read
> those ops back **as attributes** for the next fold.

## 5. Step: watch it work

The second function of [`tests/sccp.mlir`](../tests/sccp.mlir) is
Chapter 6's CSE example, but now we can *compute* it:

***tests/sccp.mlir***
```mlir
func.func @test_poly_sccp() -> !poly.poly<10> {
  %0 = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
  %p0 = poly.from_tensor %0 : tensor<3xi32> -> !poly.poly<10>
  %2 = poly.mul %p0, %p0 : !poly.poly<10>
  %3 = poly.mul %p0, %p0 : !poly.poly<10>
  %4 = poly.add %2, %3 : !poly.poly<10>
  return %2 : !poly.poly<10>
}
```

```bash
$TUTORIAL_OPT -pass-pipeline="builtin.module(func.func(sccp))" tests/sccp.mlir
```

```mlir
func.func @test_poly_sccp() -> !poly.poly<10> {
  %0 = poly.constant dense<[2, 8, 20, 24, 18]> : tensor<5xi32> : !poly.poly<10>
  %1 = poly.constant dense<[1, 4, 10, 12, 9]> : tensor<5xi32> : !poly.poly<10>
  %2 = poly.constant dense<[1, 2, 3]> : tensor<3xi32> : !poly.poly<10>
  %cst = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
  return %1 : !poly.poly<10>
}
```

Every `poly` op became a `poly.constant`, materialized by section 4's
hook. As with `arith`, the dead constants remain — sccp propagates, it
doesn't clean.

### Where does `[1, 4, 10, 12, 9]` come from?

Recall the coefficient-list convention: index *i* holds the coefficient
of xⁱ, so `[1, 2, 3]` is the polynomial **p = 1 + 2x + 3x²**, and the
`poly.mul` fold computed p·p. Multiply it out the schoolbook way — every
term of the first factor times every term of the second:

```
        ·     1      2x     3x²
      1       1      2x     3x²
      2x      2x     4x²    6x³
      3x²     3x²    6x³    9x⁴
```

Collecting like powers (the diagonals of the table):

| power | products | sum |
|-------|--------------------------|-----|
| x⁰ | 1·1 | **1** |
| x¹ | 1·2 + 2·1 | **4** |
| x² | 1·3 + 2·2 + 3·1 | **10** |
| x³ | 2·3 + 3·2 | **12** |
| x⁴ | 3·3 | **9** |

so p² = 1 + 4x + 10x² + 12x³ + 9x⁴ = `[1, 4, 10, 12, 9]`. This
"multiply everything, collect by i+j" is the **convolution** in
section 3's double loop: `result[(i + j) % degree] += lhs[i] * rhs[j]` —
with the `% degree` irrelevant here since the highest power (4) stays
below 10. Then the `poly.add` fold doubles it coefficient-wise to
`[2, 8, 20, 24, 18]`.

You can check the fold in isolation — one mul, one command:

```mlir
func.func @square() -> !poly.poly<10> {
  %coeffs = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
  %p = poly.from_tensor %coeffs : tensor<3xi32> -> !poly.poly<10>
  %sq = poly.mul %p, %p : !poly.poly<10>
  return %sq : !poly.poly<10>
}
```

```bash
$TUTORIAL_OPT --canonicalize square.mlir
```

```mlir
func.func @square() -> !poly.poly<10> {
  %0 = poly.constant dense<[1, 4, 10, 12, 9]> : tensor<5xi32> : !poly.poly<10>
  return %0 : !poly.poly<10>
}
```

And there's an independent spot-check that doesn't require trusting
either the compiler or the table: evaluate both sides at some point, say
x = 7. p(7) = 1 + 2·7 + 3·49 = 162 (Chapter 5's `%4`!), so p²(7) must be
162² = 26244; and indeed
1 + 4·7 + 10·49 + 12·343 + 9·2401 = 26244. A polynomial identity that
holds at enough points is the identity — evaluating at a random point is
a quick smoke test for any coefficient arithmetic you don't trust.

**The wraparound, verified.** Build x⁶ and square it in a degree-10 ring
(x¹² ≡ x²):

```mlir
func.func @wraparound() -> !poly.poly<10> {
  %c = arith.constant dense<[0, 0, 0, 0, 0, 0, 1]> : tensor<7xi32>
  %x6 = poly.from_tensor %c : tensor<7xi32> -> !poly.poly<10>
  %sq = poly.mul %x6, %x6 : !poly.poly<10>
  return %sq : !poly.poly<10>
}
```

```bash
$TUTORIAL_OPT --canonicalize wraparound.mlir
```

```mlir
%0 = poly.constant dense<[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : tensor<13xi32> : !poly.poly<10>
```

The `1` landed at index 2 — `(6+6) % 10` — and there is section 3's
infelicity in the flesh: a `tensor<13xi32>` (7+7−1) holding a degree-10
ring element, with indices 10–12 necessarily zero.

**A declined fold, verified.** Make the tensor a function argument instead
of a constant and the null guards fire — `--canonicalize` leaves the ops
exactly as written:

```mlir
func.func @cant_fold(%arg0: tensor<3xi32>) -> !poly.poly<10> {
  %0 = poly.from_tensor %arg0 : tensor<3xi32> -> !poly.poly<10>
  %1 = poly.mul %0, %0 : !poly.poly<10>
  return %1 : !poly.poly<10>
}
```

Returning `nullptr` is not an error; it's the ordinary "not my case"
answer, and most fold invocations end that way.

Run the test suite entry:

```bash
bazel test //tests:sccp.mlir.test                 # Bazel
llvm-lit -sv build-ninja/tests --filter sccp      # CMake
```

## 6. Beyond constants: folds that return values

Everything above returned attributes, but `OpFoldResult` is a sum type:
a fold may instead return an **existing SSA value**, expressing identities
like "this op is a no-op". Upstream examples abound:
`complex.create(complex.re(%z), complex.im(%z))` folds to `%z`;
`a - b + b` folds to `a`. The shape in code is a *match* rather than a
computation: inspect your operands' defining ops (`getDefiningOp<T>()`,
Chapter 3's tool) and return the value that makes you redundant. `poly`
has an obvious candidate — `poly.to_tensor(poly.from_tensor(%t))` is just
`%t` — which is exercise 3.

## Where to go next

The dialect now optimizes itself arithmetically. But try feeding it
nonsense — a `poly.eval` whose result type doesn't match its point type,
say — and the only guards are traits and type constraints. Custom
*semantic* checks (and that `EvalOp::verify` sitting in `PolyOps.cpp`,
plus Chapter 5's mysterious `Has32BitArguments` trait) are
[Chapter 8: Verifiers](08-verifiers.md).

**Exercises**

1. Verify the fold arithmetic by hand: square 1 + 2x + 3x² and check
   `[1, 4, 10, 12, 9]`; then predict `poly.constant dense<[0, 1]>`
   (i.e. x) raised to the 10th power in `!poly.poly<10>` — chain
   `poly.mul`s and confirm with `--canonicalize` that x¹⁰ ≡ 1.
2. Reproduce the section-5 wraparound and declined-fold experiments. Then
   improve the fold on paper: how would you change `MulOp::fold` to emit a
   `tensor<10xi32>` (trimmed to the degree bound) instead of
   `tensor<13xi32>`? What existing test would your change break?
3. Sketch `ToTensorOp::fold` implementing
   `to_tensor(from_tensor(%t)) → %t` (return a *value*, not an
   attribute — check the input's defining op). What should it return when
   the polynomial's degree bound is smaller than the original tensor?
   (That subtlety is why the repo doesn't ship this fold.)
4. Run `--sccp` followed by `--cse` on `tests/sccp.mlir` and compare with
   `--canonicalize` alone. Which dead constants does each combination
   remove, and why does canonicalize get them all?
5. `arith.constant 7` in `@test_arith_sccp` folds through `arith`'s own
   folders. Skim `arith`'s fold implementations in the upstream sources
   (`AddIOp::fold` in `ArithOps.cpp`) and find the same
   `constFoldBinaryOp` helper we used — the pattern you learned here is
   the upstream pattern.
