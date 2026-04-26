// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// pvm_slashing.cu — CUDA peer of pvm_slashing.metal.

#include "pvm_kernels_common.cuh"

namespace pvm::cuda {

extern "C" __global__ void pvm_slashing_transition(
    const PVMRoundDescriptor* desc,
    const SlashEvidence*      evidence,
    ValidatorSlot*            validators,
    SlashEvidence*            slashing,
    uint32_t*                 applied_out,
    uint64_t*                 total_slashed_out,
    uint32_t                  validator_count,
    uint32_t                  slashing_count)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    uint32_t applied = 0;
    uint64_t total_slashed = 0;
    uint32_t cursor = 0;
    uint32_t count = desc->slash_evidence_count;

    for (uint32_t i = 0; i < count; ++i) {
        const SlashEvidence& ev = evidence[i];
        uint32_t v_idx = validator_locate(validators, validator_count,
                                          ev.validator_id, false);
        if (v_idx == 0xFFFFFFFFu) continue;
        ValidatorSlot& v = validators[v_idx];
        if ((v.status & kStatusTombstoned) != 0u) continue;

        uint64_t amount = ev.slash_amount;
        if (amount == 0u) {
            switch (ev.kind) {
                case kEvEquivocation: amount = v.weight / 20u;  break;
                case kEvDowntime:     amount = v.weight / 100u; break;
                case kEvInvalidVote:  amount = v.weight / 50u;  break;
            }
        }
        if (amount > v.weight) amount = v.weight;
        v.weight = sat_sub_u64(v.weight, amount);
        total_slashed = sat_add_u64(total_slashed, amount);

        if (ev.kind == kEvEquivocation) {
            v.status |= kStatusTombstoned;
            v.status &= ~kStatusActive;
        } else {
            v.status |= kStatusJailed;
            v.status &= ~kStatusActive;
            uint32_t jail_for = ev.jail_for_epochs == 0u ? 100u : ev.jail_for_epochs;
            uint32_t until = ev.epoch + jail_for;
            if (until > v.jail_until_epoch) v.jail_until_epoch = until;
        }

        if (cursor < slashing_count) {
            SlashEvidence& dst = slashing[cursor];
            dst.validator_id     = ev.validator_id;
            dst.height           = ev.height;
            dst.slash_amount     = ev.slash_amount;
            dst.kind             = ev.kind;
            dst.epoch            = ev.epoch;
            dst.jail_for_epochs  = ev.jail_for_epochs;
            dst._pad0            = 0;
            for (uint32_t k = 0; k < 32u; ++k) dst.evidence_digest[k] = ev.evidence_digest[k];
            dst._pad1            = 0;
            ++cursor;
        }
        ++applied;
    }

    *applied_out = applied;
    *total_slashed_out = total_slashed;
}

}  // namespace pvm::cuda
