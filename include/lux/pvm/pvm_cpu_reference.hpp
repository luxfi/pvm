// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file pvm_cpu_reference.hpp
/// CPU reference implementation of the PVM transition kernels — the
/// differential-fuzz oracle for cross-backend determinism (CPU vs Metal
/// vs CUDA must produce byte-identical roots on the same input).
///
/// The reference processes input ops in canonical order:
///   1. ValidatorSetApply  (validator_ops, in order)
///   2. StakeTransition    (stake_ops, in order)
///   3. SlashingTransition (slash_evidence, in order)
///   4. EpochTransition    (closes the epoch if desc.closing_flag != 0)
///
/// The same canonical order is what pvm_*.metal / pvm_*.cu must produce.
///
/// State carries forward across run_reference() calls only via the
/// validator/stake arenas the caller threads through — the reference is
/// pure on (state, ops) and does not retain hidden state.

#pragma once

#include "pvm_gpu_layout.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace pvm::gpu::ref {

struct PVMReferenceState {
    std::vector<ValidatorSlot> validators;     ///< sized to kDefaultValidatorSlots
    std::vector<StakeRecord>   stake;          ///< sized to kDefaultStakeSlots
    std::vector<SlashEvidence> slashing;       ///< sized to kDefaultSlashSlots
    EpochState                 epoch{};

    static PVMReferenceState empty();
};

PVMTransitionResult run_reference(PVMReferenceState& state,
                                  const PVMRoundDescriptor& desc,
                                  std::span<const ValidatorOp>   validator_ops,
                                  std::span<const StakeOp>       stake_ops,
                                  std::span<const SlashEvidence> slash_evidence);

}  // namespace pvm::gpu::ref
