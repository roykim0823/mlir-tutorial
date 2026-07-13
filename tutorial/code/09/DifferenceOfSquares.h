#ifndef TUTORIAL_CODE_09_DIFFERENCEOFSQUARES_H_
#define TUTORIAL_CODE_09_DIFFERENCEOFSQUARES_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/Pass/Pass.h"

namespace mlir {
namespace tutorial {
namespace poly {

// Scaffolding specific to this frozen copy: the pattern was originally
// registered inside SubOp::getCanonicalizationPatterns, so it ran as part of
// --canonicalize. This standalone pass exposes the same pattern under its
// own flag, so it can be run without editing the Poly dialect in lib/.
class DifferenceOfSquaresPass
    : public PassWrapper<DifferenceOfSquaresPass,
                         OperationPass<mlir::func::FuncOp>> {
private:
  void runOnOperation() override;

  StringRef getArgument() const final { return "difference-of-squares"; }

  StringRef getDescription() const final {
    return "Apply Chapter 9's C++ DifferenceOfSquares pattern";
  }
};

} // namespace poly
} // namespace tutorial
} // namespace mlir

#endif // TUTORIAL_CODE_09_DIFFERENCEOFSQUARES_H_
