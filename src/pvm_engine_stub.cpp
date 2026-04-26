// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// pvm_engine_stub.cpp — CPU-only fallback for PVMGPUEngine::create().
// Linked when neither Metal nor CUDA is enabled, so tests still build.
// The stub returns nullptr, exercising the CPU-only test paths.

#include "lux/pvm/pvm_gpu_engine.hpp"

namespace pvm::gpu {

#if !defined(__APPLE__)
std::unique_ptr<PVMGPUEngine> PVMGPUEngine::create() {
    return nullptr;
}
#else
// On Apple this stub is unused (the Metal driver provides create()).
// Keep a weak fallback so the symbol exists in CPU-only Apple builds.
__attribute__((weak)) std::unique_ptr<PVMGPUEngine> PVMGPUEngine::create() {
    return nullptr;
}
#endif

}  // namespace pvm::gpu
