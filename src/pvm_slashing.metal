// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// pvm_slashing.metal — SlashingTransition kernel.
//
// Walks slash_evidence in canonical order. Equivocation tombstones;
// downtime / invalid-vote jails. Default policy slash percentages match
// the CPU reference exactly.

#include "pvm_kernels_common.h.metal"

kernel void pvm_slashing_transition(
    device const PVMRoundDescriptor* desc        [[buffer(0)]],
    device const SlashEvidence*      evidence    [[buffer(1)]],
    device ValidatorSlot*            validators  [[buffer(2)]],
    device SlashEvidence*            slashing    [[buffer(3)]],
    device atomic_uint*              applied_out [[buffer(4)]],
    device atomic_uint*              total_lo    [[buffer(5)]],
    device atomic_uint*              total_hi    [[buffer(6)]],
    constant uint&                   validator_count [[buffer(7)]],
    constant uint&                   slashing_count  [[buffer(8)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0u) return;
    uint applied = 0;
    ulong total_slashed = 0;
    uint cursor = 0;
    uint count = desc->slash_evidence_count;

    for (uint i = 0; i < count; ++i) {
        const device SlashEvidence& ev = evidence[i];
        uint v_idx = validator_locate(validators, validator_count,
                                      ev.validator_id, false);
        if (v_idx == 0xFFFFFFFFu) continue;
        device ValidatorSlot& v = validators[v_idx];
        if ((v.status & kStatusTombstoned) != 0u) continue;

        ulong amount = ev.slash_amount;
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
            uint jail_for = ev.jail_for_epochs == 0u ? 100u : ev.jail_for_epochs;
            uint until = ev.epoch + jail_for;
            if (until > v.jail_until_epoch) v.jail_until_epoch = until;
        }

        if (cursor < slashing_count) {
            device SlashEvidence& dst = slashing[cursor];
            dst.validator_id     = ev.validator_id;
            dst.height           = ev.height;
            dst.slash_amount     = ev.slash_amount;
            dst.kind             = ev.kind;
            dst.epoch            = ev.epoch;
            dst.jail_for_epochs  = ev.jail_for_epochs;
            dst._pad0            = 0;
            for (uint k = 0; k < 32u; ++k) dst.evidence_digest[k] = ev.evidence_digest[k];
            dst._pad1            = 0;
            ++cursor;
        }
        ++applied;
    }

    atomic_store_explicit(applied_out, applied, memory_order_relaxed);
    // 64-bit total split into two uint atomics (MSL has no 64-bit atomics).
    atomic_store_explicit(total_lo, (uint)(total_slashed & 0xFFFFFFFFu), memory_order_relaxed);
    atomic_store_explicit(total_hi, (uint)((total_slashed >> 32) & 0xFFFFFFFFu),
                          memory_order_relaxed);
}
