// Chapter 3's driver, pre-tablegen (the original hand-written form shown in
// chapter 3 section 3): explicit PassRegistration lines, no generated
// register*Passes() functions. See README.md in this directory.
#include "tutorial/code/03/AffineFullUnroll.h"
#include "tutorial/code/03/AffineFullUnrollPatternRewrite.h"
#include "tutorial/code/03/MulToAdd.h"
#include "mlir/include/mlir/InitAllDialects.h"
#include "mlir/include/mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);

  mlir::PassRegistration<mlir::tutorial::AffineFullUnrollPass>();
  mlir::PassRegistration<mlir::tutorial::AffineFullUnrollPassAsPatternRewrite>();
  mlir::PassRegistration<mlir::tutorial::MulToAddPass>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Tutorial Pass Driver", registry));
}
