add_mlir_library(Tutorial03Passes
    AffineFullUnroll.cpp
    AffineFullUnrollPatternRewrite.cpp
    MulToAdd.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/tutorial/code/03/

    LINK_LIBS PUBLIC
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(tutorial-opt-03 tutorial-opt.cpp)

llvm_update_compile_flags(tutorial-opt-03)
target_link_libraries(tutorial-opt-03 PRIVATE
    ${dialect_libs}
    ${conversion_libs}
    Tutorial03Passes
    MLIROptLib
    MLIRPass
)

mlir_check_all_link_libraries(tutorial-opt-03)
