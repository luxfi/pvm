# Copyright (C) 2026, Lux Partners Limited. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# PVM (Platform-VM) GPU substrate — GPU-native P-Chain transition kernels.
# v0.52 ships the layout + CPU reference oracle. v0.53 adds CUDA + Metal
# transition kernels with cross-backend determinism tests.

cmake_minimum_required(VERSION 3.20)

project(lux-pvm
    VERSION 0.52.0
    DESCRIPTION "GPU-native P-Chain transition substrate"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(LUX_PVM_BUILD_TESTS "Build PVM tests" ON)

# ---- core library (host code: layout + CPU reference) ----------------------

add_library(lux_pvm_core STATIC
    src/pvm_cpu_reference.cpp
)

target_include_directories(lux_pvm_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(lux_pvm_core PUBLIC cxx_std_20)

set_target_properties(lux_pvm_core PROPERTIES
    OUTPUT_NAME "lux_pvm_core"
    POSITION_INDEPENDENT_CODE ON
)

# ---- tests ----------------------------------------------------------------

if(LUX_PVM_BUILD_TESTS)
    enable_testing()

    add_executable(pvm-layout-test
        test/pvm_layout_test.cpp
    )
    target_link_libraries(pvm-layout-test PRIVATE lux_pvm_core)
    target_compile_features(pvm-layout-test PRIVATE cxx_std_20)
    add_test(NAME pvm-layout-test COMMAND pvm-layout-test)
endif()

# ---- summary --------------------------------------------------------------

message(STATUS "")
message(STATUS "=== Lux PVM Configuration ===")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build Tests: ${LUX_PVM_BUILD_TESTS}")
message(STATUS "")
