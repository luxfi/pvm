// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file pvm_gpu_engine.hpp
/// PVMGPUEngine — GPU-native P-Chain transition substrate.
///
/// Lifecycle (mirrors QuasarGPUEngine):
///   begin_round(PVMRoundDescriptor)
///   push_validator_ops / push_stake_ops / push_slash_evidence
///   run_epoch / run_until_done
///   poll_round_result  -> PVMTransitionResult (with epoch_root for Quasar)
///   end_round
///
/// One PVMTransitionResult.epoch_root equals the Quasar round descriptor's
/// pchain_validator_root for the same epoch — this is the linkage that
/// makes P-Chain GPU-native under LP-137.

#pragma once

#include "pvm_gpu_layout.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace pvm::gpu {

struct PVMRoundHandle {
    uint64_t opaque = 0;
    bool valid() const { return opaque != 0; }
};

class PVMGPUEngine {
public:
    virtual ~PVMGPUEngine() = default;

    static std::unique_ptr<PVMGPUEngine> create();

    virtual PVMRoundHandle begin_round(const PVMRoundDescriptor& desc) = 0;

    virtual void push_validator_ops(PVMRoundHandle h,
                                    std::span<const ValidatorOp> ops) = 0;
    virtual void push_stake_ops(PVMRoundHandle h,
                                std::span<const StakeOp> ops) = 0;
    virtual void push_slash_evidence(PVMRoundHandle h,
                                     std::span<const SlashEvidence> ev) = 0;

    virtual PVMTransitionResult run_epoch(PVMRoundHandle h) = 0;
    virtual PVMTransitionResult run_until_done(PVMRoundHandle h,
                                               std::size_t max_epochs = 64) = 0;
    virtual PVMTransitionResult poll_round_result(PVMRoundHandle h) const = 0;

    virtual void end_round(PVMRoundHandle h) = 0;

    virtual bool round_active() const = 0;
    virtual const char* device_name() const = 0;

protected:
    PVMGPUEngine() = default;
    PVMGPUEngine(const PVMGPUEngine&) = delete;
    PVMGPUEngine& operator=(const PVMGPUEngine&) = delete;
};

}  // namespace pvm::gpu
