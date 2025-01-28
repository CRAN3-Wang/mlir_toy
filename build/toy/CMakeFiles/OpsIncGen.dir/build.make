# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/crane/dev/mlir_toy

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/crane/dev/mlir_toy/build

# Utility rule file for OpsIncGen.

# Include any custom commands dependencies for this target.
include toy/CMakeFiles/OpsIncGen.dir/compiler_depend.make

# Include the progress variables for this target.
include toy/CMakeFiles/OpsIncGen.dir/progress.make

toy/CMakeFiles/OpsIncGen: toy/generated/include/Ops.h.inc
toy/CMakeFiles/OpsIncGen: toy/generated/include/Ops.h.inc
toy/CMakeFiles/OpsIncGen: toy/generated/src/Ops.cpp.inc
toy/CMakeFiles/OpsIncGen: toy/generated/src/Ops.cpp.inc
toy/CMakeFiles/OpsIncGen: toy/generated/include/Dialect.h.inc
toy/CMakeFiles/OpsIncGen: toy/generated/include/Dialect.h.inc
toy/CMakeFiles/OpsIncGen: toy/generated/src/Dialect.cpp.inc
toy/CMakeFiles/OpsIncGen: toy/generated/src/Dialect.cpp.inc

toy/generated/include/Dialect.h.inc: /home/crane/dev/mlir-tutorial/install/bin/mlir-tblgen
toy/generated/include/Dialect.h.inc: ../toy/Ops.td
toy/generated/include/Dialect.h.inc: ../toy/ToyCombine.td
toy/generated/include/Dialect.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/CodeGen/SDNodeProperties.td
toy/generated/include/Dialect.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/CodeGen/ValueTypes.td
toy/generated/include/Dialect.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Frontend/Directive/DirectiveBase.td
toy/generated/include/Dialect.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Frontend/OpenACC/ACC.td
toy/generated/include/Dialect.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Frontend/OpenMP/OMP.td
toy/generated/include/Dialect.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/Attributes.td
toy/generated/include/Dialect.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/Intrinsics.td
toy/generated/include/Dialect.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsAArch64.td
toy/generated/include/Dialect.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsAMDGPU.td
toy/generated/include/Dialect.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsARM.td
toy/generated/include/Dialect.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsBPF.td
toy/generated/include/Dialect.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsDirectX.td
toy/generated/include/Dialect.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsHexagon.td
toy/generated/include/Dialect.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsHexagonDep.td
toy/generated/include/Dialect.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsLoongArch.td
toy/generated/include/Dialect.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsMips.td
toy/generated/include/Dialect.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsNVVM.td
toy/generated/include/Dialect.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsPowerPC.td
toy/generated/include/Dialect.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsRISCV.td
toy/generated/include/Dialect.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsRISCVXTHead.td
toy/generated/include/Dialect.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsRISCVXsf.td
toy/generated/include/Dialect.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsSPIRV.td
toy/generated/include/Dialect.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsSystemZ.td
toy/generated/include/Dialect.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsVE.td
toy/generated/include/Dialect.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsVEVL.gen.td
toy/generated/include/Dialect.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsWebAssembly.td
toy/generated/include/Dialect.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsX86.td
toy/generated/include/Dialect.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsXCore.td
toy/generated/include/Dialect.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Option/OptParser.td
toy/generated/include/Dialect.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/TableGen/Automaton.td
toy/generated/include/Dialect.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/TableGen/SearchableTable.td
toy/generated/include/Dialect.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Target/GenericOpcodes.td
toy/generated/include/Dialect.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Target/GlobalISel/Combine.td
toy/generated/include/Dialect.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Target/GlobalISel/RegisterBank.td
toy/generated/include/Dialect.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Target/GlobalISel/SelectionDAGCompat.td
toy/generated/include/Dialect.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Target/GlobalISel/Target.td
toy/generated/include/Dialect.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Target/Target.td
toy/generated/include/Dialect.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Target/TargetCallingConv.td
toy/generated/include/Dialect.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Target/TargetInstrPredicate.td
toy/generated/include/Dialect.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Target/TargetItinerary.td
toy/generated/include/Dialect.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Target/TargetPfmCounters.td
toy/generated/include/Dialect.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Target/TargetSchedule.td
toy/generated/include/Dialect.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Target/TargetSelectionDAG.td
toy/generated/include/Dialect.h.inc: ../toy/Ops.td
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/crane/dev/mlir_toy/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building generated/include/Dialect.h.inc..."
	cd /home/crane/dev/mlir_toy/build/toy && /home/crane/dev/mlir-tutorial/install/bin/mlir-tblgen -gen-dialect-decls -I /home/crane/dev/mlir_toy/toy -I/home/crane/dev/mlir-tutorial/install/include -I/home/crane/dev/mlir-tutorial/install/include -I/home/crane/dev/mlir_toy/toy/include -I/home/crane/dev/mlir_toy/build/toy/generated /home/crane/dev/mlir_toy/toy/Ops.td --write-if-changed -o /home/crane/dev/mlir_toy/build/toy/generated/include/Dialect.h.inc

toy/generated/include/Ops.h.inc: /home/crane/dev/mlir-tutorial/install/bin/mlir-tblgen
toy/generated/include/Ops.h.inc: ../toy/Ops.td
toy/generated/include/Ops.h.inc: ../toy/ToyCombine.td
toy/generated/include/Ops.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/CodeGen/SDNodeProperties.td
toy/generated/include/Ops.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/CodeGen/ValueTypes.td
toy/generated/include/Ops.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Frontend/Directive/DirectiveBase.td
toy/generated/include/Ops.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Frontend/OpenACC/ACC.td
toy/generated/include/Ops.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Frontend/OpenMP/OMP.td
toy/generated/include/Ops.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/Attributes.td
toy/generated/include/Ops.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/Intrinsics.td
toy/generated/include/Ops.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsAArch64.td
toy/generated/include/Ops.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsAMDGPU.td
toy/generated/include/Ops.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsARM.td
toy/generated/include/Ops.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsBPF.td
toy/generated/include/Ops.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsDirectX.td
toy/generated/include/Ops.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsHexagon.td
toy/generated/include/Ops.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsHexagonDep.td
toy/generated/include/Ops.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsLoongArch.td
toy/generated/include/Ops.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsMips.td
toy/generated/include/Ops.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsNVVM.td
toy/generated/include/Ops.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsPowerPC.td
toy/generated/include/Ops.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsRISCV.td
toy/generated/include/Ops.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsRISCVXTHead.td
toy/generated/include/Ops.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsRISCVXsf.td
toy/generated/include/Ops.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsSPIRV.td
toy/generated/include/Ops.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsSystemZ.td
toy/generated/include/Ops.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsVE.td
toy/generated/include/Ops.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsVEVL.gen.td
toy/generated/include/Ops.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsWebAssembly.td
toy/generated/include/Ops.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsX86.td
toy/generated/include/Ops.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsXCore.td
toy/generated/include/Ops.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Option/OptParser.td
toy/generated/include/Ops.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/TableGen/Automaton.td
toy/generated/include/Ops.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/TableGen/SearchableTable.td
toy/generated/include/Ops.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Target/GenericOpcodes.td
toy/generated/include/Ops.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Target/GlobalISel/Combine.td
toy/generated/include/Ops.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Target/GlobalISel/RegisterBank.td
toy/generated/include/Ops.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Target/GlobalISel/SelectionDAGCompat.td
toy/generated/include/Ops.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Target/GlobalISel/Target.td
toy/generated/include/Ops.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Target/Target.td
toy/generated/include/Ops.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Target/TargetCallingConv.td
toy/generated/include/Ops.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Target/TargetInstrPredicate.td
toy/generated/include/Ops.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Target/TargetItinerary.td
toy/generated/include/Ops.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Target/TargetPfmCounters.td
toy/generated/include/Ops.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Target/TargetSchedule.td
toy/generated/include/Ops.h.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Target/TargetSelectionDAG.td
toy/generated/include/Ops.h.inc: ../toy/Ops.td
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/crane/dev/mlir_toy/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building generated/include/Ops.h.inc..."
	cd /home/crane/dev/mlir_toy/build/toy && /home/crane/dev/mlir-tutorial/install/bin/mlir-tblgen -gen-op-decls -I /home/crane/dev/mlir_toy/toy -I/home/crane/dev/mlir-tutorial/install/include -I/home/crane/dev/mlir-tutorial/install/include -I/home/crane/dev/mlir_toy/toy/include -I/home/crane/dev/mlir_toy/build/toy/generated /home/crane/dev/mlir_toy/toy/Ops.td --write-if-changed -o /home/crane/dev/mlir_toy/build/toy/generated/include/Ops.h.inc

toy/generated/src/Dialect.cpp.inc: /home/crane/dev/mlir-tutorial/install/bin/mlir-tblgen
toy/generated/src/Dialect.cpp.inc: ../toy/Ops.td
toy/generated/src/Dialect.cpp.inc: ../toy/ToyCombine.td
toy/generated/src/Dialect.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/CodeGen/SDNodeProperties.td
toy/generated/src/Dialect.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/CodeGen/ValueTypes.td
toy/generated/src/Dialect.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Frontend/Directive/DirectiveBase.td
toy/generated/src/Dialect.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Frontend/OpenACC/ACC.td
toy/generated/src/Dialect.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Frontend/OpenMP/OMP.td
toy/generated/src/Dialect.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/Attributes.td
toy/generated/src/Dialect.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/Intrinsics.td
toy/generated/src/Dialect.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsAArch64.td
toy/generated/src/Dialect.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsAMDGPU.td
toy/generated/src/Dialect.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsARM.td
toy/generated/src/Dialect.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsBPF.td
toy/generated/src/Dialect.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsDirectX.td
toy/generated/src/Dialect.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsHexagon.td
toy/generated/src/Dialect.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsHexagonDep.td
toy/generated/src/Dialect.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsLoongArch.td
toy/generated/src/Dialect.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsMips.td
toy/generated/src/Dialect.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsNVVM.td
toy/generated/src/Dialect.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsPowerPC.td
toy/generated/src/Dialect.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsRISCV.td
toy/generated/src/Dialect.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsRISCVXTHead.td
toy/generated/src/Dialect.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsRISCVXsf.td
toy/generated/src/Dialect.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsSPIRV.td
toy/generated/src/Dialect.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsSystemZ.td
toy/generated/src/Dialect.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsVE.td
toy/generated/src/Dialect.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsVEVL.gen.td
toy/generated/src/Dialect.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsWebAssembly.td
toy/generated/src/Dialect.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsX86.td
toy/generated/src/Dialect.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsXCore.td
toy/generated/src/Dialect.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Option/OptParser.td
toy/generated/src/Dialect.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/TableGen/Automaton.td
toy/generated/src/Dialect.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/TableGen/SearchableTable.td
toy/generated/src/Dialect.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Target/GenericOpcodes.td
toy/generated/src/Dialect.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Target/GlobalISel/Combine.td
toy/generated/src/Dialect.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Target/GlobalISel/RegisterBank.td
toy/generated/src/Dialect.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Target/GlobalISel/SelectionDAGCompat.td
toy/generated/src/Dialect.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Target/GlobalISel/Target.td
toy/generated/src/Dialect.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Target/Target.td
toy/generated/src/Dialect.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Target/TargetCallingConv.td
toy/generated/src/Dialect.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Target/TargetInstrPredicate.td
toy/generated/src/Dialect.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Target/TargetItinerary.td
toy/generated/src/Dialect.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Target/TargetPfmCounters.td
toy/generated/src/Dialect.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Target/TargetSchedule.td
toy/generated/src/Dialect.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Target/TargetSelectionDAG.td
toy/generated/src/Dialect.cpp.inc: ../toy/Ops.td
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/crane/dev/mlir_toy/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building generated/src/Dialect.cpp.inc..."
	cd /home/crane/dev/mlir_toy/build/toy && /home/crane/dev/mlir-tutorial/install/bin/mlir-tblgen -gen-dialect-defs -I /home/crane/dev/mlir_toy/toy -I/home/crane/dev/mlir-tutorial/install/include -I/home/crane/dev/mlir-tutorial/install/include -I/home/crane/dev/mlir_toy/toy/include -I/home/crane/dev/mlir_toy/build/toy/generated /home/crane/dev/mlir_toy/toy/Ops.td --write-if-changed -o /home/crane/dev/mlir_toy/build/toy/generated/src/Dialect.cpp.inc

toy/generated/src/Ops.cpp.inc: /home/crane/dev/mlir-tutorial/install/bin/mlir-tblgen
toy/generated/src/Ops.cpp.inc: ../toy/Ops.td
toy/generated/src/Ops.cpp.inc: ../toy/ToyCombine.td
toy/generated/src/Ops.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/CodeGen/SDNodeProperties.td
toy/generated/src/Ops.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/CodeGen/ValueTypes.td
toy/generated/src/Ops.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Frontend/Directive/DirectiveBase.td
toy/generated/src/Ops.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Frontend/OpenACC/ACC.td
toy/generated/src/Ops.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Frontend/OpenMP/OMP.td
toy/generated/src/Ops.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/Attributes.td
toy/generated/src/Ops.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/Intrinsics.td
toy/generated/src/Ops.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsAArch64.td
toy/generated/src/Ops.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsAMDGPU.td
toy/generated/src/Ops.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsARM.td
toy/generated/src/Ops.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsBPF.td
toy/generated/src/Ops.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsDirectX.td
toy/generated/src/Ops.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsHexagon.td
toy/generated/src/Ops.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsHexagonDep.td
toy/generated/src/Ops.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsLoongArch.td
toy/generated/src/Ops.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsMips.td
toy/generated/src/Ops.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsNVVM.td
toy/generated/src/Ops.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsPowerPC.td
toy/generated/src/Ops.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsRISCV.td
toy/generated/src/Ops.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsRISCVXTHead.td
toy/generated/src/Ops.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsRISCVXsf.td
toy/generated/src/Ops.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsSPIRV.td
toy/generated/src/Ops.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsSystemZ.td
toy/generated/src/Ops.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsVE.td
toy/generated/src/Ops.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsVEVL.gen.td
toy/generated/src/Ops.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsWebAssembly.td
toy/generated/src/Ops.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsX86.td
toy/generated/src/Ops.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/IR/IntrinsicsXCore.td
toy/generated/src/Ops.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Option/OptParser.td
toy/generated/src/Ops.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/TableGen/Automaton.td
toy/generated/src/Ops.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/TableGen/SearchableTable.td
toy/generated/src/Ops.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Target/GenericOpcodes.td
toy/generated/src/Ops.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Target/GlobalISel/Combine.td
toy/generated/src/Ops.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Target/GlobalISel/RegisterBank.td
toy/generated/src/Ops.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Target/GlobalISel/SelectionDAGCompat.td
toy/generated/src/Ops.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Target/GlobalISel/Target.td
toy/generated/src/Ops.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Target/Target.td
toy/generated/src/Ops.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Target/TargetCallingConv.td
toy/generated/src/Ops.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Target/TargetInstrPredicate.td
toy/generated/src/Ops.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Target/TargetItinerary.td
toy/generated/src/Ops.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Target/TargetPfmCounters.td
toy/generated/src/Ops.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Target/TargetSchedule.td
toy/generated/src/Ops.cpp.inc: /home/crane/dev/mlir-tutorial/install/include/llvm/Target/TargetSelectionDAG.td
toy/generated/src/Ops.cpp.inc: ../toy/Ops.td
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/crane/dev/mlir_toy/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building generated/src/Ops.cpp.inc..."
	cd /home/crane/dev/mlir_toy/build/toy && /home/crane/dev/mlir-tutorial/install/bin/mlir-tblgen -gen-op-defs -I /home/crane/dev/mlir_toy/toy -I/home/crane/dev/mlir-tutorial/install/include -I/home/crane/dev/mlir-tutorial/install/include -I/home/crane/dev/mlir_toy/toy/include -I/home/crane/dev/mlir_toy/build/toy/generated /home/crane/dev/mlir_toy/toy/Ops.td --write-if-changed -o /home/crane/dev/mlir_toy/build/toy/generated/src/Ops.cpp.inc

OpsIncGen: toy/CMakeFiles/OpsIncGen
OpsIncGen: toy/generated/include/Dialect.h.inc
OpsIncGen: toy/generated/include/Ops.h.inc
OpsIncGen: toy/generated/src/Dialect.cpp.inc
OpsIncGen: toy/generated/src/Ops.cpp.inc
OpsIncGen: toy/CMakeFiles/OpsIncGen.dir/build.make
.PHONY : OpsIncGen

# Rule to build all files generated by this target.
toy/CMakeFiles/OpsIncGen.dir/build: OpsIncGen
.PHONY : toy/CMakeFiles/OpsIncGen.dir/build

toy/CMakeFiles/OpsIncGen.dir/clean:
	cd /home/crane/dev/mlir_toy/build/toy && $(CMAKE_COMMAND) -P CMakeFiles/OpsIncGen.dir/cmake_clean.cmake
.PHONY : toy/CMakeFiles/OpsIncGen.dir/clean

toy/CMakeFiles/OpsIncGen.dir/depend:
	cd /home/crane/dev/mlir_toy/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/crane/dev/mlir_toy /home/crane/dev/mlir_toy/toy /home/crane/dev/mlir_toy/build /home/crane/dev/mlir_toy/build/toy /home/crane/dev/mlir_toy/build/toy/CMakeFiles/OpsIncGen.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : toy/CMakeFiles/OpsIncGen.dir/depend

