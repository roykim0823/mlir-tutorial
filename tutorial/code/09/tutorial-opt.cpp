// Driver for Chapter 9's preserved C++ DifferenceOfSquares pattern
// (scaffolding around the frozen pattern). See README.md in this directory.
#include "lib/Dialect/Poly/PolyDialect.h"
#include "tutorial/code/09/DifferenceOfSquares.h"
#include "mlir/include/mlir/InitAllDialects.h"
#include "mlir/include/mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::tutorial::poly::PolyDialect>();
  mlir::registerAllDialects(registry);

  mlir::PassRegistration<mlir::tutorial::poly::DifferenceOfSquaresPass>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Tutorial Pass Driver", registry));
}
