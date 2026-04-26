# Copyright (C) 2026, Lux Partners Limited. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# PVM (Platform-VM) GPU substrate.
# v0.52: layout + CPU reference oracle.
# v0.53: CUDA + Metal transition kernels with cross-backend determinism.

cmake_minimum_required(VERSION 3.20)

project(lux-pvm
    VERSION 0.53.0
    DESCRIPTION "GPU-native P-Chain transition substrate"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(LUX_PVM_BUILD_TESTS "Build PVM tests" ON)
option(LUX_PVM_ENABLE_METAL "Build Metal backend (Apple)" ${APPLE})
option(LUX_PVM_ENABLE_CUDA  "Build CUDA backend" OFF)

# ---- core library (host code: layout + CPU reference) ---------------------

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

# ---- Metal driver (Apple) -------------------------------------------------

if(LUX_PVM_ENABLE_METAL AND APPLE)
    add_library(lux_pvm_metal STATIC
        src/pvm_gpu_engine.mm
    )
    target_include_directories(lux_pvm_metal PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    )
    target_compile_features(lux_pvm_metal PUBLIC cxx_std_20)
    target_link_libraries(lux_pvm_metal PUBLIC lux_pvm_core
        "-framework Metal" "-framework Foundation")
    set_target_properties(lux_pvm_metal PROPERTIES
        OUTPUT_NAME "lux_pvm_metal"
        POSITION_INDEPENDENT_CODE ON
    )

    # Stage the .metal sources for any source-based runtime compile fallback.
    foreach(src
        pvm_kernels_common.h.metal
        pvm_validator_set.metal
        pvm_staking.metal
        pvm_slashing.metal
        pvm_transition.metal)
        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/src/${src}
            ${CMAKE_CURRENT_BINARY_DIR}/src/${src}
            COPYONLY)
    endforeach()

    # Pre-compile a unified .metallib so the runtime can skip the online
    # MSL compiler entirely (more reliable than newLibraryWithSource for
    # large kernels). The combined source is the same one the runtime
    # would build by concatenation: common header + four kernels with
    # the include-line stripped.
    set(PVM_METAL_COMBINED ${CMAKE_CURRENT_BINARY_DIR}/src/pvm_combined.metal)
    add_custom_command(
        OUTPUT ${PVM_METAL_COMBINED}
        COMMAND bash -c "cat ${CMAKE_CURRENT_SOURCE_DIR}/src/pvm_kernels_common.h.metal > ${PVM_METAL_COMBINED} && for f in pvm_validator_set.metal pvm_staking.metal pvm_slashing.metal pvm_transition.metal; do grep -v 'pvm_kernels_common.h.metal' ${CMAKE_CURRENT_SOURCE_DIR}/src/$f >> ${PVM_METAL_COMBINED}; done"
        DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/src/pvm_kernels_common.h.metal
            ${CMAKE_CURRENT_SOURCE_DIR}/src/pvm_validator_set.metal
            ${CMAKE_CURRENT_SOURCE_DIR}/src/pvm_staking.metal
            ${CMAKE_CURRENT_SOURCE_DIR}/src/pvm_slashing.metal
            ${CMAKE_CURRENT_SOURCE_DIR}/src/pvm_transition.metal
        VERBATIM)

    set(PVM_METAL_AIR ${CMAKE_CURRENT_BINARY_DIR}/src/pvm_combined.air)
    set(PVM_METAL_LIB ${CMAKE_CURRENT_BINARY_DIR}/src/pvm.metallib)

    # NOTE on -O0: the staking kernel's inner reward-distribution loop
    # combined with its switch-over-StakeOpKind triggers a PSO compile
    # crash in Metal's Apple-Silicon optimizer pipeline (the AOT AIR
    # compiles fine at -Os; the runtime PSO conversion fails with
    # XPC_ERROR_CONNECTION_INTERRUPTED after retries). The kernels are
    # 1x1 single-thread dispatches, so we lose nothing by skipping the
    # optimizer here. Reproducible isolation: compile at -Os and the
    # PSO for pvm_stake_transition fails; -O0 always succeeds.
    add_custom_command(
        OUTPUT ${PVM_METAL_AIR}
        COMMAND xcrun -sdk macosx metal -std=metal3.0 -O0
            -Wno-pragma-once-outside-header
            -Wno-unused-const-variable
            -c ${PVM_METAL_COMBINED} -o ${PVM_METAL_AIR}
        DEPENDS ${PVM_METAL_COMBINED}
        VERBATIM)

    add_custom_command(
        OUTPUT ${PVM_METAL_LIB}
        COMMAND xcrun -sdk macosx metallib ${PVM_METAL_AIR} -o ${PVM_METAL_LIB}
        DEPENDS ${PVM_METAL_AIR}
        VERBATIM)

    add_custom_target(pvm_metallib ALL DEPENDS ${PVM_METAL_LIB})
    add_dependencies(lux_pvm_metal pvm_metallib)
endif()

# ---- CUDA driver ---------------------------------------------------------

if(LUX_PVM_ENABLE_CUDA)
    enable_language(CUDA)
    add_library(lux_pvm_cuda STATIC
        src/pvm_gpu_engine_cuda.cpp
        src/pvm_cuda_launchers.cu
        src/pvm_validator_set.cu
        src/pvm_staking.cu
        src/pvm_slashing.cu
        src/pvm_transition.cu
    )
    target_include_directories(lux_pvm_cuda PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    )
    set_target_properties(lux_pvm_cuda PROPERTIES
        CUDA_STANDARD 17
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        OUTPUT_NAME "lux_pvm_cuda"
    )
    target_link_libraries(lux_pvm_cuda PUBLIC lux_pvm_core)
endif()

# ---- tests ----------------------------------------------------------------

if(LUX_PVM_BUILD_TESTS)
    enable_testing()

    # Layout test — host-only.
    add_executable(pvm-layout-test test/pvm_layout_test.cpp)
    target_link_libraries(pvm-layout-test PRIVATE lux_pvm_core)
    target_compile_features(pvm-layout-test PRIVATE cxx_std_20)
    add_test(NAME pvm-layout-test COMMAND pvm-layout-test)

    # GPU engine test — Apple-only (.mm).
    if(LUX_PVM_ENABLE_METAL AND APPLE)
        add_executable(pvm-gpu-engine-test test/pvm_gpu_engine_test.mm)
        target_link_libraries(pvm-gpu-engine-test PRIVATE lux_pvm_metal)
        target_compile_features(pvm-gpu-engine-test PRIVATE cxx_std_20)
        add_test(NAME pvm-gpu-engine-test COMMAND pvm-gpu-engine-test)
    endif()

    # Cross-backend determinism — links to whichever GPU backend is enabled.
    add_executable(pvm-determinism-test test/pvm_determinism_test.cpp)
    target_compile_features(pvm-determinism-test PRIVATE cxx_std_20)
    if(LUX_PVM_ENABLE_METAL AND APPLE)
        target_link_libraries(pvm-determinism-test PRIVATE lux_pvm_metal)
    elseif(LUX_PVM_ENABLE_CUDA)
        target_link_libraries(pvm-determinism-test PRIVATE lux_pvm_cuda)
    else()
        # CPU-only path — provide a stub create() so the test still links.
        target_sources(pvm-determinism-test PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/pvm_engine_stub.cpp)
        target_link_libraries(pvm-determinism-test PRIVATE lux_pvm_core)
    endif()
    add_test(NAME pvm-determinism-test COMMAND pvm-determinism-test)
endif()

message(STATUS "")
message(STATUS "=== Lux PVM Configuration ===")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build Tests: ${LUX_PVM_BUILD_TESTS}")
message(STATUS "  Metal:       ${LUX_PVM_ENABLE_METAL}")
message(STATUS "  CUDA:        ${LUX_PVM_ENABLE_CUDA}")
message(STATUS "")
