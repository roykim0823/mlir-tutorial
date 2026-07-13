// Preserved Chapter 3 code, pre-tablegen (chapter 3 section 7); the
// current version is lib/Transform/Affine. See README.md in this directory.
#ifndef TUTORIAL_CODE_03_AFFINEFULLUNROLLPATTERNREWRITE_H_
#define TUTORIAL_CODE_03_AFFINEFULLUNROLLPATTERNREWRITE_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/Pass/Pass.h"

namespace mlir {
namespace tutorial {

class AffineFullUnrollPassAsPatternRewrite
    : public PassWrapper<AffineFullUnrollPassAsPatternRewrite,
                         OperationPass<mlir::func::FuncOp>> {
private:
  void runOnOperation() override;

  StringRef getArgument() const final { return "affine-full-unroll-rewrite"; }

  StringRef getDescription() const final {
    return "Fully unroll all affine loops using the pattern rewrite engine";
  }
};

} // namespace tutorial
} // namespace mlir

#endif // TUTORIAL_CODE_03_AFFINEFULLUNROLLPATTERNREWRITE_H_
