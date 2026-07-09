add_mlir_library(Tutorial09DifferenceOfSquares
    DifferenceOfSquares.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/tutorial/code/09/

    DEPENDS
    MLIRPolyOpsIncGen

    LINK_LIBS PUBLIC
    MLIRPoly
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(tutorial-opt-09 tutorial-opt.cpp)

llvm_update_compile_flags(tutorial-opt-09)
target_link_libraries(tutorial-opt-09 PRIVATE
    ${dialect_libs}
    ${conversion_libs}
    Tutorial09DifferenceOfSquares
    MLIRPoly
    MLIROptLib
    MLIRPass
)

mlir_check_all_link_libraries(tutorial-opt-09)
