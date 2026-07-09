#ifndef TUTORIAL_CODE_03_MULTOADD_H_
#define TUTORIAL_CODE_03_MULTOADD_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/Pass/Pass.h"

namespace mlir {
namespace tutorial {

class MulToAddPass
    : public PassWrapper<MulToAddPass, OperationPass<mlir::func::FuncOp>> {
private:
  void runOnOperation() override;

  StringRef getArgument() const final { return "mul-to-add"; }

  StringRef getDescription() const final {
    return "Convert multiplications to repeated additions";
  }
};

} // namespace tutorial
} // namespace mlir

#endif // TUTORIAL_CODE_03_MULTOADD_H_
