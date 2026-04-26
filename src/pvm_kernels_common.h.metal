// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// pvm_kernels_common.h.metal — shared device code for the four PVM kernels.
//
// Layout structs MUST match pvm_gpu_layout.hpp byte-for-byte. keccak256 is
// the same Keccak-f[1600] / 0x01 / 0x80 padding used by the CPU reference
// (pvm_cpu_reference.cpp) and the cevm/quasar reference. Determinism
// across CPU / Metal / CUDA hinges on this exact byte-for-byte recipe.

#pragma once

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Layout structs — must match pvm_gpu_layout.hpp byte-for-byte.
// =============================================================================

struct alignas(16) ValidatorSlot {
    ulong  validator_id;          // 0
    ulong  weight;                // 8
    uchar  bls_pubkey[48];        // 16
    uchar  ringtail_pubkey[32];   // 64
    uchar  mldsa_pubkey[32];      // 96
    uchar  mldsa_groth16_root[32];// 128
    uint   status;                // 160
    uint   jail_until_epoch;      // 164
    uint   occupied;              // 168
    uint   _pad0;                 // 172  -> 176
};

struct alignas(16) StakeRecord {
    ulong  delegator_id;
    ulong  validator_id;
    ulong  amount;
    ulong  lock_until_epoch;
    ulong  reward_accumulator;
    uint   commission_bps;
    uint   status;
    uint   epoch_bonded;
    uint   epoch_unbonded;
    ulong  _pad0;                 // -> 64
};

struct alignas(16) SlashEvidence {
    ulong  validator_id;
    ulong  height;
    ulong  slash_amount;
    uint   kind;
    uint   epoch;
    uint   jail_for_epochs;
    uint   _pad0;
    uchar  evidence_digest[32];
    ulong  _pad1;                 // -> 80
};

struct alignas(16) EpochState {
    ulong  current_epoch;
    ulong  next_epoch_height;
    ulong  total_active_stake;
    uint   active_validator_count;
    uint   pending_drop_count;
    uchar  validator_set_root[32];
    uchar  stake_root[32];
    uchar  slashing_root[32];
    uchar  epoch_root[32];
};

struct alignas(16) PVMRoundDescriptor {
    ulong  chain_id;
    ulong  round;
    ulong  timestamp_ns;
    ulong  epoch;
    uint   mode;
    uint   validator_op_count;
    uint   stake_op_count;
    uint   slash_evidence_count;
    uint   closing_flag;
    uint   _pad0;
    ulong  _pad1;
    uchar  parent_epoch_root[32];
};

struct alignas(16) ValidatorOp {
    ulong  validator_id;
    ulong  weight;
    uchar  bls_pubkey[48];
    uchar  ringtail_pubkey[32];
    uchar  mldsa_pubkey[32];
    uchar  mldsa_groth16_root[32];
    uint   kind;
    uint   jail_until_epoch;
    uint   epoch;
    uint   _pad0;
};

struct alignas(16) StakeOp {
    ulong  delegator_id;
    ulong  validator_id;
    ulong  amount;
    ulong  lock_until_epoch;
    ulong  source_validator_id;
    uint   kind;
    uint   commission_bps;
    uint   epoch;
    uint   _pad0;
    ulong  _pad1;                 // -> 64
};

struct alignas(16) PVMTransitionResult {
    uint   status;
    uint   validator_apply_count;
    uint   stake_apply_count;
    uint   slash_apply_count;
    uint   active_validator_count;
    uint   pending_drop_count;
    uint   jailed_count;
    uint   tombstoned_count;
    ulong  total_active_stake;
    ulong  total_slashed;
    ulong  total_rewards;
    ulong  epoch;
    uchar  validator_set_root[32];
    uchar  stake_root[32];
    uchar  slashing_root[32];
    uchar  epoch_root[32];
};

// =============================================================================
// Status / kind constants — match pvm_gpu_layout.hpp / pvm_cpu_reference.cpp
// =============================================================================

constant uint kStatusActive       = 0x1u;
constant uint kStatusJailed       = 0x2u;
constant uint kStatusTombstoned   = 0x4u;
constant uint kStatusPendingAdd   = 0x8u;
constant uint kStatusPendingDrop  = 0x10u;

constant uint kStakeStatusActive    = 1u;
constant uint kStakeStatusUnbonding = 2u;
constant uint kStakeStatusRetired   = 3u;

constant uint kVOpAdd          = 0u;
constant uint kVOpRemove       = 1u;
constant uint kVOpUpdateWeight = 2u;
constant uint kVOpJail         = 3u;
constant uint kVOpUnjail       = 4u;
constant uint kVOpRotateKeys   = 5u;

constant uint kSOpBond         = 0u;
constant uint kSOpUnbond       = 1u;
constant uint kSOpDelegate     = 2u;
constant uint kSOpRedelegate   = 3u;
constant uint kSOpReward       = 4u;
constant uint kSOpCommission   = 5u;

constant uint kEvEquivocation  = 0u;
constant uint kEvDowntime      = 1u;
constant uint kEvInvalidVote   = 2u;

constant uint kModeValidator   = 0u;
constant uint kModeStake       = 1u;
constant uint kModeSlashing    = 2u;
constant uint kModeEpoch       = 3u;
constant uint kModeFullRound   = 4u;

constant ulong kRewardScale = 1000000000000000000UL;  // 1e18

// =============================================================================
// keccak256 — bit-identical to pvm_cpu_reference.cpp
// =============================================================================

constant ulong kKeccakRC[24] = {
    0x0000000000000001UL, 0x0000000000008082UL,
    0x800000000000808AUL, 0x8000000080008000UL,
    0x000000000000808BUL, 0x0000000080000001UL,
    0x8000000080008081UL, 0x8000000000008009UL,
    0x000000000000008AUL, 0x0000000000000088UL,
    0x0000000080008009UL, 0x000000008000000AUL,
    0x000000008000808BUL, 0x800000000000008BUL,
    0x8000000000008089UL, 0x8000000000008003UL,
    0x8000000000008002UL, 0x8000000000000080UL,
    0x000000000000800AUL, 0x800000008000000AUL,
    0x8000000080008081UL, 0x8000000000008080UL,
    0x0000000080000001UL, 0x8000000080008008UL,
};

constant uint kKeccakRot[25] = {
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14,
};

// rotl64 is called with n=0 in the rho-pi step (kKeccakRot[0] == 0).
// `x >> (64 - n)` is UB at n=0; mask the shift count to make it well-defined.
// The n==0 case yields `x | x == x` which is the intended identity.
inline ulong rotl64(ulong x, uint n) {
    return (x << (n & 63u)) | (x >> ((64u - n) & 63u));
}

inline void keccak_f1600(thread ulong* s) {
    for (uint round = 0; round < 24u; ++round) {
        ulong c[5];
        for (uint x = 0; x < 5u; ++x)
            c[x] = s[x] ^ s[x+5] ^ s[x+10] ^ s[x+15] ^ s[x+20];
        ulong d[5];
        for (uint x = 0; x < 5u; ++x)
            d[x] = c[(x + 4u) % 5u] ^ rotl64(c[(x + 1u) % 5u], 1u);
        for (uint y = 0; y < 25u; y += 5u)
            for (uint x = 0; x < 5u; ++x)
                s[y + x] ^= d[x];
        ulong b[25];
        for (uint y = 0; y < 5u; ++y)
            for (uint x = 0; x < 5u; ++x) {
                uint i = x + 5u * y;
                uint j = y + 5u * ((2u * x + 3u * y) % 5u);
                b[j] = rotl64(s[i], kKeccakRot[i]);
            }
        for (uint y = 0; y < 25u; y += 5u) {
            ulong t0 = b[y+0], t1 = b[y+1], t2 = b[y+2], t3 = b[y+3], t4 = b[y+4];
            s[y+0] = t0 ^ ((~t1) & t2);
            s[y+1] = t1 ^ ((~t2) & t3);
            s[y+2] = t2 ^ ((~t3) & t4);
            s[y+3] = t3 ^ ((~t4) & t0);
            s[y+4] = t4 ^ ((~t0) & t1);
        }
        s[0] ^= kKeccakRC[round];
    }
}

inline void keccak256(thread const uchar* data, ulong len, thread uchar* out) {
    ulong s[25] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    const uint rate = 136u;
    ulong off = 0;
    while (len - off >= rate) {
        for (uint i = 0; i < rate; ++i) {
            uint lane = i / 8u, sh = (i % 8u) * 8u;
            s[lane] ^= ((ulong)data[off + i]) << sh;
        }
        keccak_f1600(s);
        off += rate;
    }
    uchar block[136] = {};
    ulong rem = len - off;
    for (ulong i = 0; i < rem; ++i) block[i] = data[off + i];
    block[rem]      ^= 0x01;
    block[rate - 1] ^= 0x80;
    for (uint i = 0; i < rate; ++i) {
        uint lane = i / 8u, sh = (i % 8u) * 8u;
        s[lane] ^= ((ulong)block[i]) << sh;
    }
    keccak_f1600(s);
    for (uint i = 0; i < 32u; ++i) {
        uint lane = i / 8u, sh = (i % 8u) * 8u;
        out[i] = (uchar)((s[lane] >> sh) & 0xFFu);
    }
}

inline void absorb_u32(thread uchar* dst, uint off, uint v) {
    for (uint k = 0; k < 4u; ++k) dst[off + k] = (uchar)((v >> (k*8u)) & 0xFFu);
}

inline void absorb_u64(thread uchar* dst, uint off, ulong v) {
    for (uint k = 0; k < 8u; ++k) dst[off + k] = (uchar)((v >> (k*8u)) & 0xFFu);
}

inline ulong sat_add_u64(ulong a, ulong b) {
    ulong r = a + b;
    return (r < a) ? 0xFFFFFFFFFFFFFFFFUL : r;
}

inline ulong sat_sub_u64(ulong a, ulong b) {
    return (a < b) ? 0u : (a - b);
}

// =============================================================================
// Validator / stake-record open-addressing locators (same hash as CPU ref)
// =============================================================================

inline uint validator_index_hash(ulong validator_id, uint mask) {
    ulong h = 0xcbf29ce484222325UL;
    h = (h ^ validator_id) * 0x100000001b3UL;
    return (uint)h & mask;
}

inline uint validator_locate(device ValidatorSlot* tab, uint count,
                             ulong validator_id, bool insert_if_missing)
{
    uint mask = count - 1u;
    uint idx  = validator_index_hash(validator_id, mask);
    for (uint probe = 0; probe < count; ++probe) {
        device ValidatorSlot& s = tab[idx];
        if (s.occupied == 0u) {
            if (insert_if_missing) {
                s.validator_id = validator_id;
                s.weight = 0;
                s.status = 0;
                s.jail_until_epoch = 0;
                s.occupied = 1u;
                for (uint k = 0; k < 48u; ++k) s.bls_pubkey[k] = 0;
                for (uint k = 0; k < 32u; ++k) s.ringtail_pubkey[k] = 0;
                for (uint k = 0; k < 32u; ++k) s.mldsa_pubkey[k] = 0;
                for (uint k = 0; k < 32u; ++k) s.mldsa_groth16_root[k] = 0;
                return idx;
            }
            return 0xFFFFFFFFu;
        }
        if (s.validator_id == validator_id) return idx;
        idx = (idx + 1u) & mask;
    }
    return 0xFFFFFFFFu;
}

inline uint stake_record_index_hash(ulong delegator, ulong validator, uint mask) {
    ulong composite = delegator ^ (validator + 0x9E3779B97F4A7C15UL +
                                   (delegator << 6) + (delegator >> 2));
    return (uint)composite & mask;
}

inline uint stake_record_locate(device StakeRecord* tab, uint count,
                                ulong delegator, ulong validator,
                                bool insert_if_missing)
{
    uint mask = count - 1u;
    uint idx  = stake_record_index_hash(delegator, validator, mask);
    for (uint probe = 0; probe < count; ++probe) {
        device StakeRecord& s = tab[idx];
        if (s.status == 0u) {
            if (insert_if_missing) {
                s.delegator_id = delegator;
                s.validator_id = validator;
                s.amount = 0;
                s.lock_until_epoch = 0;
                s.reward_accumulator = 0;
                s.commission_bps = 0;
                s.status = kStakeStatusActive;
                s.epoch_bonded = 0;
                s.epoch_unbonded = 0;
                return idx;
            }
            return 0xFFFFFFFFu;
        }
        if (s.delegator_id == delegator && s.validator_id == validator) return idx;
        idx = (idx + 1u) & mask;
    }
    return 0xFFFFFFFFu;
}
