// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// pvm_transition.metal — EpochTransition kernel.
//
// Promotes pending_add -> active, drops pending_drop -> tombstoned,
// auto-unjails expired jail windows, then composes the four roots:
//   validator_set_root  =  fold-keccak over occupied ValidatorSlot leaves
//   stake_root          =  fold-keccak over non-empty StakeRecord leaves
//   slashing_root       =  fold-keccak over non-empty SlashEvidence leaves
//   epoch_root          =  keccak(parent || vroot || sroot || slroot ||
//                                 epoch || total_stake || active_count)
//
// epoch_root is the value the Quasar round descriptor binds as
// pchain_validator_root — this is the linkage that completes LP-137
// GPU-native compliance for the P-Chain.

#include "pvm_kernels_common.h.metal"

kernel void pvm_epoch_transition(
    device const PVMRoundDescriptor* desc           [[buffer(0)]],
    device ValidatorSlot*            validators     [[buffer(1)]],
    device StakeRecord*              stake          [[buffer(2)]],
    device SlashEvidence*            slashing       [[buffer(3)]],
    device EpochState*               epoch          [[buffer(4)]],
    device PVMTransitionResult*      result         [[buffer(5)]],
    constant uint&                   validator_count [[buffer(6)]],
    constant uint&                   stake_count     [[buffer(7)]],
    constant uint&                   slashing_count  [[buffer(8)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0u) return;

    // -- promotion / auto-unjail --
    uint pending_drop = 0;
    ulong target_epoch = (desc->closing_flag != 0u) ? desc->epoch + 1u : desc->epoch;
    for (uint i = 0; i < validator_count; ++i) {
        device ValidatorSlot& s = validators[i];
        if (s.occupied == 0u) continue;
        if ((s.status & kStatusPendingAdd) != 0u) {
            s.status &= ~kStatusPendingAdd;
        }
        if ((s.status & kStatusPendingDrop) != 0u) {
            s.status &= ~kStatusPendingDrop;
            s.status |= kStatusTombstoned;
            ++pending_drop;
        }
        if ((s.status & kStatusJailed) != 0u
            && s.jail_until_epoch != 0u
            && (uint)target_epoch >= s.jail_until_epoch
            && (s.status & kStatusTombstoned) == 0u) {
            s.status &= ~kStatusJailed;
            s.status |= kStatusActive;
            s.jail_until_epoch = 0;
        }
    }
    epoch->pending_drop_count = pending_drop;

    // -- validator_set_root + counts --
    uchar acc[32]; for (uint k = 0; k < 32u; ++k) acc[k] = 0;
    uint active = 0, jailed = 0, tombstoned = 0;
    ulong total_stake = 0;
    for (uint i = 0; i < validator_count; ++i) {
        device ValidatorSlot& s = validators[i];
        if (s.occupied == 0u) continue;
        if ((s.status & kStatusTombstoned) != 0u) ++tombstoned;
        if ((s.status & kStatusJailed) != 0u) ++jailed;
        if ((s.status & kStatusActive) != 0u) {
            ++active;
            total_stake = sat_add_u64(total_stake, s.weight);
        }
        uchar leaf[8 + 8 + 4 + 4 + 48 + 32 + 32 + 32 + 4];
        uint o = 0;
        absorb_u64(leaf, o, s.validator_id); o += 8;
        absorb_u64(leaf, o, s.weight);       o += 8;
        absorb_u32(leaf, o, s.status);       o += 4;
        absorb_u32(leaf, o, s.jail_until_epoch); o += 4;
        for (uint k = 0; k < 48u; ++k) leaf[o + k] = s.bls_pubkey[k]; o += 48;
        for (uint k = 0; k < 32u; ++k) leaf[o + k] = s.ringtail_pubkey[k]; o += 32;
        for (uint k = 0; k < 32u; ++k) leaf[o + k] = s.mldsa_pubkey[k]; o += 32;
        for (uint k = 0; k < 32u; ++k) leaf[o + k] = s.mldsa_groth16_root[k]; o += 32;
        absorb_u32(leaf, o, i); o += 4;

        uchar leaf_hash[32];
        keccak256(leaf, o, leaf_hash);

        uchar buf[64];
        for (uint k = 0; k < 32u; ++k) buf[k] = acc[k];
        for (uint k = 0; k < 32u; ++k) buf[32 + k] = leaf_hash[k];
        keccak256(buf, 64, acc);
    }
    for (uint k = 0; k < 32u; ++k) epoch->validator_set_root[k] = acc[k];

    // -- stake_root + total_rewards --
    for (uint k = 0; k < 32u; ++k) acc[k] = 0;
    ulong total_rewards = 0;
    for (uint i = 0; i < stake_count; ++i) {
        device StakeRecord& s = stake[i];
        if (s.status == 0u) continue;
        total_rewards = sat_add_u64(total_rewards, s.reward_accumulator);

        uchar leaf[8 + 8 + 8 + 8 + 8 + 4 + 4 + 4 + 4 + 4];
        uint o = 0;
        absorb_u64(leaf, o, s.delegator_id);       o += 8;
        absorb_u64(leaf, o, s.validator_id);       o += 8;
        absorb_u64(leaf, o, s.amount);             o += 8;
        absorb_u64(leaf, o, s.lock_until_epoch);   o += 8;
        absorb_u64(leaf, o, s.reward_accumulator); o += 8;
        absorb_u32(leaf, o, s.commission_bps);     o += 4;
        absorb_u32(leaf, o, s.status);             o += 4;
        absorb_u32(leaf, o, s.epoch_bonded);       o += 4;
        absorb_u32(leaf, o, s.epoch_unbonded);     o += 4;
        absorb_u32(leaf, o, i);                    o += 4;

        uchar leaf_hash[32];
        keccak256(leaf, o, leaf_hash);
        uchar buf[64];
        for (uint k = 0; k < 32u; ++k) buf[k] = acc[k];
        for (uint k = 0; k < 32u; ++k) buf[32 + k] = leaf_hash[k];
        keccak256(buf, 64, acc);
    }
    for (uint k = 0; k < 32u; ++k) epoch->stake_root[k] = acc[k];

    // -- slashing_root --
    for (uint k = 0; k < 32u; ++k) acc[k] = 0;
    for (uint i = 0; i < slashing_count; ++i) {
        device SlashEvidence& ev = slashing[i];
        // detect zero slot
        bool zero_digest = true;
        for (uint k = 0; k < 32u; ++k) if (ev.evidence_digest[k] != 0) { zero_digest = false; break; }
        if (ev.validator_id == 0u && zero_digest && ev.height == 0u) continue;

        uchar leaf[8 + 8 + 8 + 4 + 4 + 4 + 32 + 4];
        uint o = 0;
        absorb_u64(leaf, o, ev.validator_id);    o += 8;
        absorb_u64(leaf, o, ev.height);          o += 8;
        absorb_u64(leaf, o, ev.slash_amount);    o += 8;
        absorb_u32(leaf, o, ev.kind);            o += 4;
        absorb_u32(leaf, o, ev.epoch);           o += 4;
        absorb_u32(leaf, o, ev.jail_for_epochs); o += 4;
        for (uint k = 0; k < 32u; ++k) leaf[o + k] = ev.evidence_digest[k]; o += 32;
        absorb_u32(leaf, o, i);                  o += 4;

        uchar leaf_hash[32];
        keccak256(leaf, o, leaf_hash);
        uchar buf[64];
        for (uint k = 0; k < 32u; ++k) buf[k] = acc[k];
        for (uint k = 0; k < 32u; ++k) buf[32 + k] = leaf_hash[k];
        keccak256(buf, 64, acc);
    }
    for (uint k = 0; k < 32u; ++k) epoch->slashing_root[k] = acc[k];

    // -- epoch metadata --
    epoch->active_validator_count = active;
    epoch->total_active_stake = total_stake;
    if (desc->closing_flag != 0u) {
        epoch->current_epoch = target_epoch;
    }

    // -- composed epoch_root --
    uchar composed[32 + 32 + 32 + 32 + 8 + 8 + 4];
    uint o = 0;
    for (uint k = 0; k < 32u; ++k) composed[o + k] = desc->parent_epoch_root[k]; o += 32;
    for (uint k = 0; k < 32u; ++k) composed[o + k] = epoch->validator_set_root[k]; o += 32;
    for (uint k = 0; k < 32u; ++k) composed[o + k] = epoch->stake_root[k];        o += 32;
    for (uint k = 0; k < 32u; ++k) composed[o + k] = epoch->slashing_root[k];     o += 32;
    absorb_u64(composed, o, epoch->current_epoch);     o += 8;
    absorb_u64(composed, o, epoch->total_active_stake); o += 8;
    absorb_u32(composed, o, epoch->active_validator_count); o += 4;

    uchar epoch_root_local[32];
    keccak256(composed, o, epoch_root_local);
    for (uint k = 0; k < 32u; ++k) epoch->epoch_root[k] = epoch_root_local[k];

    // -- write result --
    for (uint k = 0; k < 32u; ++k) result->validator_set_root[k] = epoch->validator_set_root[k];
    for (uint k = 0; k < 32u; ++k) result->stake_root[k]         = epoch->stake_root[k];
    for (uint k = 0; k < 32u; ++k) result->slashing_root[k]      = epoch->slashing_root[k];
    for (uint k = 0; k < 32u; ++k) result->epoch_root[k]         = epoch->epoch_root[k];
    result->active_validator_count = active;
    result->jailed_count           = jailed;
    result->tombstoned_count       = tombstoned;
    result->total_active_stake     = total_stake;
    result->total_rewards          = total_rewards;
    result->pending_drop_count     = pending_drop;
    result->epoch                  = epoch->current_epoch;
    result->status                 = 1u;
}
