# Chapter 10: Dialect Conversion

Nine chapters in, `poly` is a complete high-level IR — and completely
unable to run. This chapter starts the descent: **lowering** `poly` into
upstream dialects (`arith`, `tensor`, `scf`), so that
[Chapter 11](11-lowering-through-llvm.md) can
carry it the rest of the way to an executable. On the way we meet the one
piece of pass machinery Chapter 3 couldn't teach, because it only
matters when *types* change: the **dialect conversion framework**.

Everything lives in
[`lib/Conversion/PolyToStandard/`](../lib/Conversion/PolyToStandard/) —
the `Conversion/` directory promised by Chapter 3 §2's layout tour —
and runs as `--poly-to-standard`.

**What you will learn:**

- Why type changes make lowering fundamentally harder than rewriting —
  and how the conversion framework solves it.
- The four moving parts: `TypeConverter`, `OpConversionPattern`,
  `ConversionTarget`, and `applyPartialConversion`.
- How each `poly` op lowers — including polynomial multiplication as a
  loop nest and evaluation as Horner's method, with real output.
- Why `func.func` needs special treatment (types hide in signatures).
- How to read the framework's signature error — "failed to legalize
  unresolved materialization" — with a live reproduction.

**Prerequisites:** [Chapter 9](09-canonicalizers-and-drr.md) (all of
`poly`), [Chapter 3](03-writing-our-first-pass.md) (rewrite patterns),
and Chapter 2 §8's pipeline vocabulary. Same build setup
(`$TUTORIAL_OPT`).

---

## 1. Concepts: what makes lowering hard

Here is the plan, and it sounds like a Chapter 3 exercise: rewrite
`poly.add` into `arith.addi`, `poly.mul` into loops, and so on. As this
codebase's author puts it: *if not for types, dialect conversion would be
essentially the same as a normal pass*.

But types ruin everything. Our `TypeConverter` will say a
`!poly.poly<10>` becomes a `tensor<10xi32>` (ten 32-bit coefficients —
Chapter 5 §1's representation, made literal). Now rewrite one op:

```mlir
%2 = poly.add %0, %1 : !poly.poly<10>      // becomes...
%2 = arith.addi %0, %1 : tensor<10xi32>    // ...this
```

The moment you do, the IR is inconsistent: `%0` and `%1` are still
`!poly.poly<10>` (their defining ops aren't converted yet), so the new
`arith.addi` is ill-typed; and every not-yet-converted *user* of `%2`
still expects a `!poly.poly<10>`. Any conversion order you pick leaves a
frontier of type mismatches — and Chapter 8 taught us mismatches don't
survive a verifier. A plain greedy rewrite (Chapter 3) has no answer;
you'd need every pattern to defensively cast, and patterns would have to
fire in dependency order.

The **dialect conversion framework**
([official docs](https://mlir.llvm.org/docs/DialectConversion/)) exists to
manage that frontier. The idea:

- You declare which types change and how (**TypeConverter**).
- You write patterns that see their operands *as if already converted*
  (**OpConversionPattern** — the framework rewrites the graph in a sorted
  order and hands each pattern the in-progress converted values).
- Type mismatches on the frontier are patched with temporary
  [`builtin.unrealized_conversion_cast`](https://mlir.llvm.org/docs/Dialects/Builtin/#builtinunrealized_conversion_cast-unrealizedconversioncastop)
  ops — essentially a forced type coercion, the internal stand-in for a
  type conflict (you met these in Chapter 2 §6's pipeline — now you know
  who makes them) — that the framework removes as conversion completes.
- You declare what "done" means (**ConversionTarget**), and the framework
  checks it — any op still illegal at the end is an error.

That last point is the second big idea: **legality**. A conversion pass
doesn't just rewrite opportunistically; it has a *contract* — "after me,
no `poly` ops remain" — and the framework enforces it.

## 2. The pass shell and the TypeConverter

The pass is declared in
[`PolyToStandard.td`](../lib/Conversion/PolyToStandard/PolyToStandard.td)
(Chapter 4 machinery — note the four `dependentDialects`: the pass
*creates* `arith`, `tensor`, and `scf` ops, so those dialects must be
loaded — Chapter 4 §2's rule, now with real stakes). The interesting
code is in
[`PolyToStandard.cpp`](../lib/Conversion/PolyToStandard/PolyToStandard.cpp),
starting with the type rule. A `TypeConverter` is the first of section
1's four moving parts — the object that "declares which types change and
how" — and using one amounts to subclassing it and registering one
callback per type rule via `addConversion`:

***lib/Conversion/PolyToStandard/PolyToStandard.cpp*** (excerpt)
```cpp
class PolyToStandardTypeConverter : public TypeConverter {
 public:
  PolyToStandardTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([ctx](PolynomialType type) -> Type {
      int degreeBound = type.getDegreeBound();
      IntegerType elementTy =
          IntegerType::get(ctx, 32, IntegerType::SignednessSemantics::Signless);
      return RankedTensorType::get({degreeBound}, elementTy);
    });
  }
};
```

Two `addConversion` calls, tried in reverse order of registration: the
`PolynomialType` one maps `!poly.poly<N>` → `tensor<Nxi32>` (the degree
bound *from the type parameter* becomes the static tensor length — this
is where Chapter 5's decision to put N in the type pays off); the
identity conversion says every other type is fine as-is. Note the
signless `i32` — Chapter 8 §2's distinction, consciously chosen.

The repo's comment at this spot is worth reading in the source: no
custom *materializations* are registered. Materialization hooks tell the
framework how to build real ops bridging old→new types when a mismatch
must persist; because this lowering converts everything in one pass, the
temporary `unrealized_conversion_cast`s all cancel out by the end, and no
custom hook is needed. (Section 6 shows what happens when they *don't*
cancel. A version-drift note if you go exploring these hooks: their
upstream API has shifted over the years — argument materializations were
merged into source materializations in later LLVM releases.)

## 3. Conversion patterns

A conversion pattern is Chapter 3's rewrite pattern re-based onto the
framework: still one class per op, still a `matchAndRewrite` whose job
is to build replacement ops and replace the root — but it subclasses
`OpConversionPattern` instead of `OpRewritePattern`, and that changes
the method's signature. Here's the add lowering, in full:

***lib/Conversion/PolyToStandard/PolyToStandard.cpp*** (excerpt)
```cpp
struct ConvertAdd : public OpConversionPattern<AddOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      AddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    arith::AddIOp addOp = rewriter.create<arith::AddIOp>(
        op.getLoc(), adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op.getOperation(), addOp);
    return success();
  }
};
```

Both differences sit in that signature, and they carry the whole
framework:

- **The `OpAdaptor` parameter** (an alias for `AddOp::Adaptor`, generated
  code you met in Chapter 7 as `FoldAdaptor`'s sibling). Its accessors —
  `adaptor.getLhs()` — return the *already-type-converted* operands: by
  the time this runs, they are `tensor<10xi32>` values, no matter what
  the original IR looked like. The `op` parameter still has the original
  (poly-typed) operands, for when you need to inspect pre-conversion
  facts. That split is the framework's central service: patterns never
  see the inconsistent frontier.
- **`ConversionPatternRewriter`** — a `PatternRewriter` with extra
  conversion-aware methods. Rule of thumb: inside a conversion pattern,
  do *everything* through it, even more strictly than Chapter 3's
  "technically not allowed" comment warned.

Why does `arith.addi` accept tensors at all? Because it's
`ElementwiseMappable` — Chapter 6's trait, on the *other* side of the
trade. Polynomial addition is coefficient-wise addition, so the whole
lowering is one op. `ConvertSub` is identical with `subi`;
`ConvertToTensor` is even better — `to_tensor` becomes *nothing*
(`rewriter.replaceOp(op, adaptor.getInput())`): after conversion a
polynomial already *is* its coefficient tensor. `ConvertFromTensor` only
has work when the input tensor is shorter than the degree bound, in
which case it zero-pads with `tensor.pad` (visible in section 5's test).

### Multiplication: creating a loop nest

`poly.mul` is a cyclic convolution (Chapter 7 §3 computed it at compile
time; now we emit code to do it at *runtime*). The pattern —
`ConvertMul` in the source — builds Chapter 7's double loop as IR. It's
long but made of three readable pieces:

1. an all-zeros result tensor: `arith.constant dense<0> : tensor<Nxi32>`;
2. loop scaffolding:
   [`scf.for`](https://mlir.llvm.org/docs/Dialects/SCFDialect/#scffor-scfforop)
   ops built with `b.create<scf::ForOp>`,
   with the accumulating tensor threaded through `iter_args` (Chapter
   1 §4's idiom, now built programmatically — the lambda you pass
   `create<ForOp>` receives the loop induction variable and loop state
   and fills the body);
3. the body: extract both coefficients, multiply, add into
   `result[(i + j) mod N]` — `tensor.extract`, `arith.remui`,
   `tensor.insert`.

One style note you'll use forever: `ImplicitLocOpBuilder b(op.getLoc(),
rewriter);` wraps the rewriter so you stop passing the location to every
`create` — builders with 10+ creates get much cleaner.

Run it (all output in this chapter is real, from a degree-4 example to
keep it readable):

```mlir
func.func @lower_mul(%p: !poly.poly<4>, %q: !poly.poly<4>) -> !poly.poly<4> {
  %0 = poly.mul %p, %q : !poly.poly<4>
  return %0 : !poly.poly<4>
}
```

```bash
$TUTORIAL_OPT --poly-to-standard lower_mul.mlir
```

```mlir
func.func @lower_mul(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  %cst = arith.constant dense<0> : tensor<4xi32>
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %0 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %cst) -> (tensor<4xi32>) {
    %1 = scf.for %arg4 = %c0 to %c4 step %c1 iter_args(%arg5 = %arg3) -> (tensor<4xi32>) {
      %2 = arith.addi %arg2, %arg4 : index
      %3 = arith.remui %2, %c4 : index
      %extracted = tensor.extract %arg0[%arg2] : tensor<4xi32>
      %extracted_0 = tensor.extract %arg1[%arg4] : tensor<4xi32>
      %4 = arith.muli %extracted, %extracted_0 : i32
      %extracted_1 = tensor.extract %arg5[%3] : tensor<4xi32>
      %5 = arith.addi %4, %extracted_1 : i32
      %inserted = tensor.insert %5 into %arg5[%3] : tensor<4xi32>
      scf.yield %inserted : tensor<4xi32>
    }
    scf.yield %1 : tensor<4xi32>
  }
  return %0 : tensor<4xi32>
}
```

Read the inner body against Chapter 7's fold —
`result[(i + j) % degree] += lhs[i] * rhs[j]` — it's the same algorithm,
emitted instead of executed. (A "functional tensors" note: `tensor.insert`
doesn't mutate; it yields a *new* tensor value, which is why the
accumulator threads through `iter_args`. Turning this into actual
in-place memory is Chapter 11's
[bufferization](https://mlir.llvm.org/docs/Bufferization/) step — one of
the harder lowering pipelines in upstream MLIR, and part of why the
dialect conversion framework carries the complexity it does.)

### Evaluation: Horner's method

`ConvertEval` lowers `poly.eval` to the classic trick for evaluating a
polynomial with N multiplies instead of N²:

> accum = 0; for i = 1..N: accum = point·accum + coeff[N−i]

i.e. c₀ + x(c₁ + x(c₂ + ⋯)) evaluated inside-out. Lowered (real output):

```mlir
func.func @lower_eval(%arg0: tensor<10xi32>, %arg1: i32) -> i32 {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c11 = arith.constant 11 : index
  %c0_i32 = arith.constant 0 : i32
  %0 = scf.for %arg2 = %c1 to %c11 step %c1 iter_args(%arg3 = %c0_i32) -> (i32) {
    %1 = arith.subi %c10, %arg2 : index
    %2 = arith.muli %arg1, %arg3 : i32
    %extracted = tensor.extract %arg0[%1] : tensor<10xi32>
    %3 = arith.addi %2, %extracted : i32
    scf.yield %3 : i32
  }
  return %0 : i32
}
```

One loop, one multiply per iteration, coefficients read
highest-to-lowest via the `subi`. (Note what's *missing*: complex-typed
points. This lowering only handles the i32 case — a gap with
consequences in section 6.)

### Constants: lower by delegation

`ConvertConstant` is the cleverest three lines in the pass: it rewrites
`poly.constant` into `arith.constant` **+ `poly.from_tensor`** — a *poly*
op! That's legal mid-conversion: the framework simply keeps converting,
and the freshly created `from_tensor` is picked up by
`ConvertFromTensor`. Lowerings can be compositional; you don't have to
re-implement your own dialect's plumbing in every pattern.

## 4. Legality: the ConversionTarget

The pass body (`runOnOperation`) is where section 1's moving parts
meet: it states the legality contract as a `ConversionTarget`, collects
section 3's patterns into a `RewritePatternSet` (constructed with the
type converter), and hands the lot to `applyPartialConversion`, the
framework's driver:

***lib/Conversion/PolyToStandard/PolyToStandard.cpp*** (excerpt)
```cpp
ConversionTarget target(*context);
target.addIllegalDialect<PolyDialect>();

RewritePatternSet patterns(context);
PolyToStandardTypeConverter typeConverter(context);
patterns.add<ConvertAdd, ConvertConstant, ConvertSub, ConvertEval,
             ConvertMul, ConvertFromTensor, ConvertToTensor>(typeConverter,
                                                             context);
...
if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
  signalPassFailure();
}
```

`addIllegalDialect<PolyDialect>()` is the contract: when the dust
settles, zero `poly` ops. The framework drives patterns *toward* that
goal and fails the pass if it can't get there — unlike Chapter 3's
greedy driver, which happily stops at any fixed point.

Then comes everyone's first stumble: **ops you don't own carry your
types**. Our test functions have
signatures like `(%arg0: !poly.poly<10>)` — the type hides inside
`func.func`'s *attributes* (remember Chapter 1 §3's generic form:
`function_type` is just data). No `poly` op is involved, so our patterns
never fire, but the contract isn't met until those signatures change
too. Upstream provides drop-in helpers, one per structural-op family:

***lib/Conversion/PolyToStandard/PolyToStandard.cpp*** (excerpt)
```cpp
populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
    patterns, typeConverter);
target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
  return typeConverter.isSignatureLegal(op.getFunctionType()) &&
         typeConverter.isLegal(&op.getBody());
});
```

plus the same pairing for `func.return`, `func.call`, and
branch-interface ops. The pattern-half rewrites signatures through the
type converter; the legality-half (`addDynamicallyLegalOp`) says a
`func.func` is legal *only if* its signature contains no
convertible-but-unconverted types. Dynamic legality is the third
legality mode after legal/illegal-by-dialect — a callback deciding per
op. If you write a conversion with a new type, budget a few lines for
each structural dialect your types can flow through; forgetting one is
the classic source of section 6's error.

Finally, `applyPartialConversion` vs `applyFullConversion`: *full*
requires every op in the module be legal afterward; *partial* only
requires that no *illegal* op remain, leaving unknown ops alone. The
advice this repo adopts: partial is a strict
generalization, use it (and it produces far better errors).

## 5. Step: run the conversion

[`tests/poly_to_standard.mlir`](../tests/poly_to_standard.mlir) has one
function per pattern. The whole file converts with:

```bash
$TUTORIAL_OPT --poly-to-standard tests/poly_to_standard.mlir
```

Highlights of the real output — add becomes one op, with the *signature*
converted too (section 4's machinery at work):

```mlir
func.func @test_lower_add(%arg0: tensor<10xi32>, %arg1: tensor<10xi32>) -> tensor<10xi32> {
  %0 = arith.addi %arg0, %arg1 : tensor<10xi32>
  return %0 : tensor<10xi32>
}
```

`to_tensor`/`from_tensor` at matching sizes vanish entirely:

```mlir
func.func @test_lower_to_tensor(%arg0: tensor<10xi32>) -> tensor<10xi32> {
  return %arg0 : tensor<10xi32>
}
```

and the size-extending `from_tensor` case shows the zero-pad
(`tensor<10xi32>` into a `!poly.poly<20>`):

```mlir
%padded = tensor.pad %arg0 low[0] high[10] { ... tensor.yield %c0_i32 ... }
    : tensor<10xi32> to tensor<20xi32>
```

Run the suite entry:

```bash
bazel test //tests:poly_to_standard.mlir.test              # Bazel
llvm-lit -sv build-ninja/tests --filter poly_to_standard   # CMake
```

## 6. Step: read a legality failure

Every conversion author meets this error; better to meet it on purpose.
`ConvertEval` doesn't handle complex points (section 3), so feed it one:

```mlir
func.func @f(%p: !poly.poly<10>, %z: complex<f64>) -> complex<f64> {
  %0 = poly.eval %p, %z : (!poly.poly<10>, complex<f64>) -> complex<f64>
  return %0 : complex<f64>
}
```

```
error: failed to legalize unresolved materialization from ('i32') to
('complex<f64>') that remained live after conversion
note: see current operation:
    %5 = "builtin.unrealized_conversion_cast"(%4) : (i32) -> complex<f64>
note: see existing live user here: "func.return"(%5) : (complex<f64>) -> ()
```

Decoding it: `ConvertEval` fired anyway and produced its i32 Horner loop;
the framework patched the resulting i32-vs-complex mismatch with an
`unrealized_conversion_cast` (section 1's frontier mechanism); at
end-of-conversion the cast was still there — no pattern or
materialization could "realize" it — so the contract failed, and the
error shows you the surviving cast *and its user*. When you see this in
your own work, the question to ask is: which pattern produced a value of
the wrong type, or which structural op did I forget to populate patterns
for? (Also try `--debug` for the framework's step-by-step legalization
log — Chapter 3 exercise 3's flag, at its most useful here. And note the
pattern *should* have refused to match a complex eval with
`return failure()` — as written it happily miscompiles-then-fails; making
it decline politely is exercise 3.)

## Where to go next

We're one flight down the staircase: `poly` programs are now `tensor` +
`scf` + `arith` programs. But those still aren't executable — tensors
must become memory (bufferization), loops must become branches, and
everything must reach the `llvm` dialect and beyond. That's
[Chapter 11: Lowering through LLVM](11-lowering-through-llvm.md) —
where the `--poly-to-llvm` mega-pipeline in `tutorial-opt.cpp` (the one
piece of that file we've never discussed) finally runs, and a real
`main.c` calls a compiled polynomial evaluator.

**Exercises**

1. Reproduce section 6's failure, then run it again with `--debug` and
   find the moment the framework inserts the
   `unrealized_conversion_cast`.
2. Predict, then verify: what does
   `poly.constant dense<[1,2,3]> : tensor<3xi32> : !poly.poly<10>`
   lower to? (Chase it through `ConvertConstant` → `ConvertFromTensor`:
   how does the 3-vs-10 size mismatch resolve?)
3. Fix `ConvertEval`'s manners on paper: add the early
   `return failure()` for non-i32 points. What does the *pass* then do
   with a complex eval — and is that outcome (a surviving illegal
   `poly.eval` and a cleaner error) actually better? What would a real
   fix — lowering complex evals to `complex` dialect ops — need from the
   TypeConverter?
4. The mul lowering is O(N²) with a `remui` in the hot loop. Sketch (IR
   on paper, not C++) a version that splits the inner loop into the
   no-wraparound prefix and the wraparound suffix, eliminating `remui`.
   Which Chapter 3 pass would you use to check your hand-derived IR
   computes the same thing? (Hint: Chapter 11 gives you a better tool —
   running it.)
5. Section 4 said `partial` beats `full` conversion. Change one line in
   your head: with `applyFullConversion`, what *additional* ops in
   `tests/poly_to_standard.mlir` would need legality declarations, and
   what would break first?
