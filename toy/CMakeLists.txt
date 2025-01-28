set(LLVM_LINK_COMPONENTS Support)

include_directories(${CMAKE_SOURCE_DIR}/toy/include)

set(GENERATED_OUTPUT_DIR "generated")
set(HEADER_OUTPUT_DIR "generated/include")
set(SOURCE_OUTPUT_DIR "generated/src")
# message(STATUS "TableGen'ed code will be written to: ${HEADER_OUTPUT_DIR} and ${SOURCE_OUTPUT_DIR}")

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${HEADER_OUTPUT_DIR})
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${SOURCE_OUTPUT_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/${GENERATED_OUTPUT_DIR})

set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/Ops.td)
message(STATUS "TableGen input file: ${LLVM_TARGET_DEFINITIONS}")

mlir_tablegen(${HEADER_OUTPUT_DIR}/Ops.h.inc -gen-op-decls)
mlir_tablegen(${SOURCE_OUTPUT_DIR}/Ops.cpp.inc -gen-op-defs)
mlir_tablegen(${HEADER_OUTPUT_DIR}/Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(${SOURCE_OUTPUT_DIR}/Dialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(OpsIncGen)

set(LLVM_TARGET_DEFINITIONS ToyCombine.td)
mlir_tablegen(${GENERATED_OUTPUT_DIR}/ToyCombine.inc -gen-rewriters)
add_public_tablegen_target(ToyCombineIncGen)

add_executable(toyc src/toyc.cpp src/AST.cpp src/Dialect.cpp src/MLIRGen.cpp src/ToyCombine.cpp)
add_dependencies(toyc OpsIncGen)
add_dependencies(toyc ToyCombineIncGen)

llvm_update_compile_flags(toyc)
target_link_libraries(
  toyc
  PRIVATE MLIRAnalysis
          MLIRIR
          MLIRParser
          MLIRPass
          MLIRFunctionInterfaces
          MLIRSideEffectInterfaces
          MLIRTransforms)

mlir_check_link_libraries(toyc)