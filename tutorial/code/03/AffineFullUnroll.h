#ifndef TUTORIAL_CODE_03_AFFINEFULLUNROLL_H_
#define TUTORIAL_CODE_03_AFFINEFULLUNROLL_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/Pass/Pass.h"

namespace mlir {
namespace tutorial {

class AffineFullUnrollPass
    : public PassWrapper<AffineFullUnrollPass,
                         OperationPass<mlir::func::FuncOp>> {
private:
  void runOnOperation() override;

  StringRef getArgument() const final { return "affine-full-unroll"; }

  StringRef getDescription() const final {
    return "Fully unroll all affine loops";
  }
};

} // namespace tutorial
} // namespace mlir

#endif // TUTORIAL_CODE_03_AFFINEFULLUNROLL_H_
