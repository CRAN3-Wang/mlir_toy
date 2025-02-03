set(LLVM_LINK_COMPONENTS Support)

include_directories(${CMAKE_SOURCE_DIR}/toy/include)

set(GENERATED_OUTPUT_DIR "generated")
set(HEADER_OUTPUT_DIR "generated/include")
set(SOURCE_OUTPUT_DIR "generated/src")
# message(STATUS "TableGen'ed code will be written to: ${HEADER_OUTPUT_DIR} and ${SOURCE_OUTPUT_DIR}")

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${HEADER_OUTPUT_DIR})
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${SOURCE_OUTPUT_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/${GENERATED_OUTPUT_DIR})

set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(${HEADER_OUTPUT_DIR}/Ops.h.inc -gen-op-decls)
mlir_tablegen(${SOURCE_OUTPUT_DIR}/Ops.cpp.inc -gen-op-defs)
mlir_tablegen(${HEADER_OUTPUT_DIR}/Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(${SOURCE_OUTPUT_DIR}/Dialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(OpsIncGen)

set(LLVM_TARGET_DEFINITIONS ToyCombine.td)
mlir_tablegen(${GENERATED_OUTPUT_DIR}/ToyCombine.inc -gen-rewriters)
add_public_tablegen_target(ToyCombineIncGen)

set(LLVM_TARGET_DEFINITIONS ShapeInferenceInterface.td)
mlir_tablegen(${HEADER_OUTPUT_DIR}/ShapeInferenceOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(${SOURCE_OUTPUT_DIR}/ShapeInferenceOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(ShapeInferenceInterfaceIncGen)

add_executable(toyc 
  src/toyc.cpp 
  src/AST.cpp 
  src/Dialect.cpp 
  src/MLIRGen.cpp 
  src/ToyCombine.cpp 
  src/ShapeInferencePass.cpp
  src/LowerToAffineLoops.cpp)

add_dependencies(toyc OpsIncGen)
add_dependencies(toyc ToyCombineIncGen)
add_dependencies(toyc ShapeInferenceInterfaceIncGen)

llvm_update_compile_flags(toyc)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

target_link_libraries(
  toyc
  PRIVATE ${dialect_libs}
          ${extension_libs}
          MLIRAnalysis
          MLIRIR
          MLIRParser
          MLIRPass
          MLIRFunctionInterfaces
          MLIRSideEffectInterfaces
          MLIRTransforms
          MLIRCastInterfaces)

mlir_check_link_libraries(toyc)