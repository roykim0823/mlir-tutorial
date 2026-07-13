#include "tutorial/code/09/DifferenceOfSquares.h"

#include "lib/Dialect/Poly/PolyOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace tutorial {
namespace poly {

// The C++ version of the DifferenceOfSquares pattern, as Chapter 9 section 3
// presents it (the original form; the repo's current version is the DRR
// pattern in lib/Dialect/Poly/PolyPatterns.td).
struct DifferenceOfSquares : public OpRewritePattern<SubOp> {
  DifferenceOfSquares(mlir::MLIRContext *context)
      : OpRewritePattern<SubOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(SubOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getOperand(0);
    Value rhs = op.getOperand(1);
    if (!lhs.hasOneUse() || !rhs.hasOneUse()) {
      return failure();
    }

    auto rhsMul = rhs.getDefiningOp<MulOp>();
    auto lhsMul = lhs.getDefiningOp<MulOp>();
    if (!rhsMul || !lhsMul) {
      return failure();
    }

    bool rhsMulOpsAgree = rhsMul.getLhs() == rhsMul.getRhs();
    bool lhsMulOpsAgree = lhsMul.getLhs() == lhsMul.getRhs();
    if (!rhsMulOpsAgree || !lhsMulOpsAgree) {
      return failure();
    }

    auto x = lhsMul.getLhs();
    auto y = rhsMul.getLhs();

    AddOp newAdd = rewriter.create<AddOp>(op.getLoc(), x, y);
    SubOp newSub = rewriter.create<SubOp>(op.getLoc(), x, y);
    MulOp newMul = rewriter.create<MulOp>(op.getLoc(), newAdd, newSub);

    // The original version wrote `replaceOp(op, {newMul})`; the braced form
    // is ambiguous (ValueRange vs. Operation*) in current MLIR.
    rewriter.replaceOp(op, newMul);
    return success();
  }
};

void DifferenceOfSquaresPass::runOnOperation() {
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<DifferenceOfSquares>(&getContext());
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

} // namespace poly
} // namespace tutorial
} // namespace mlir
