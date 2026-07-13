// Preserved Chapter 3 code, pre-tablegen (chapter 3 section 5); the
// current version is lib/Transform/Affine. See README.md in this directory.
#include "tutorial/code/03/AffineFullUnroll.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"

namespace mlir {
namespace tutorial {

using mlir::affine::AffineForOp;
using mlir::affine::loopUnrollFull;

// A pass that manually walks the IR
void AffineFullUnrollPass::runOnOperation() {
  getOperation().walk([&](AffineForOp op) {
    if (failed(loopUnrollFull(op))) {
      op.emitError("unrolling failed");
      signalPassFailure();
    }
  });
}

} // namespace tutorial
} // namespace mlir
