// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// pvm_kernels_common.cuh — shared device code for the four CUDA PVM kernels.
// Layout MUST match pvm_gpu_layout.hpp byte-for-byte.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace pvm::cuda {

struct alignas(16) ValidatorSlot {
    uint64_t validator_id;
    uint64_t weight;
    uint8_t  bls_pubkey[48];
    uint8_t  ringtail_pubkey[32];
    uint8_t  mldsa_pubkey[32];
    uint8_t  mldsa_groth16_root[32];
    uint32_t status;
    uint32_t jail_until_epoch;
    uint32_t occupied;
    uint32_t _pad0;
};

struct alignas(16) StakeRecord {
    uint64_t delegator_id;
    uint64_t validator_id;
    uint64_t amount;
    uint64_t lock_until_epoch;
    uint64_t reward_accumulator;
    uint32_t commission_bps;
    uint32_t status;
    uint32_t epoch_bonded;
    uint32_t epoch_unbonded;
    uint64_t _pad0;
};

struct alignas(16) SlashEvidence {
    uint64_t validator_id;
    uint64_t height;
    uint64_t slash_amount;
    uint32_t kind;
    uint32_t epoch;
    uint32_t jail_for_epochs;
    uint32_t _pad0;
    uint8_t  evidence_digest[32];
    uint64_t _pad1;
};

struct alignas(16) EpochState {
    uint64_t current_epoch;
    uint64_t next_epoch_height;
    uint64_t total_active_stake;
    uint32_t active_validator_count;
    uint32_t pending_drop_count;
    uint8_t  validator_set_root[32];
    uint8_t  stake_root[32];
    uint8_t  slashing_root[32];
    uint8_t  epoch_root[32];
};

struct alignas(16) PVMRoundDescriptor {
    uint64_t chain_id;
    uint64_t round;
    uint64_t timestamp_ns;
    uint64_t epoch;
    uint32_t mode;
    uint32_t validator_op_count;
    uint32_t stake_op_count;
    uint32_t slash_evidence_count;
    uint32_t closing_flag;
    uint32_t _pad0;
    uint64_t _pad1;
    uint8_t  parent_epoch_root[32];
};

struct alignas(16) ValidatorOp {
    uint64_t validator_id;
    uint64_t weight;
    uint8_t  bls_pubkey[48];
    uint8_t  ringtail_pubkey[32];
    uint8_t  mldsa_pubkey[32];
    uint8_t  mldsa_groth16_root[32];
    uint32_t kind;
    uint32_t jail_until_epoch;
    uint32_t epoch;
    uint32_t _pad0;
};

struct alignas(16) StakeOp {
    uint64_t delegator_id;
    uint64_t validator_id;
    uint64_t amount;
    uint64_t lock_until_epoch;
    uint64_t source_validator_id;
    uint32_t kind;
    uint32_t commission_bps;
    uint32_t epoch;
    uint32_t _pad0;
    uint64_t _pad1;
};

struct alignas(16) PVMTransitionResult {
    uint32_t status;
    uint32_t validator_apply_count;
    uint32_t stake_apply_count;
    uint32_t slash_apply_count;
    uint32_t active_validator_count;
    uint32_t pending_drop_count;
    uint32_t jailed_count;
    uint32_t tombstoned_count;
    uint64_t total_active_stake;
    uint64_t total_slashed;
    uint64_t total_rewards;
    uint64_t epoch;
    uint8_t  validator_set_root[32];
    uint8_t  stake_root[32];
    uint8_t  slashing_root[32];
    uint8_t  epoch_root[32];
};

constexpr uint32_t kStatusActive       = 0x1u;
constexpr uint32_t kStatusJailed       = 0x2u;
constexpr uint32_t kStatusTombstoned   = 0x4u;
constexpr uint32_t kStatusPendingAdd   = 0x8u;
constexpr uint32_t kStatusPendingDrop  = 0x10u;

constexpr uint32_t kStakeStatusActive    = 1u;
constexpr uint32_t kStakeStatusUnbonding = 2u;
constexpr uint32_t kStakeStatusRetired   = 3u;

constexpr uint32_t kVOpAdd          = 0u;
constexpr uint32_t kVOpRemove       = 1u;
constexpr uint32_t kVOpUpdateWeight = 2u;
constexpr uint32_t kVOpJail         = 3u;
constexpr uint32_t kVOpUnjail       = 4u;
constexpr uint32_t kVOpRotateKeys   = 5u;

constexpr uint32_t kSOpBond         = 0u;
constexpr uint32_t kSOpUnbond       = 1u;
constexpr uint32_t kSOpDelegate     = 2u;
constexpr uint32_t kSOpRedelegate   = 3u;
constexpr uint32_t kSOpReward       = 4u;
constexpr uint32_t kSOpCommission   = 5u;

constexpr uint32_t kEvEquivocation  = 0u;
constexpr uint32_t kEvDowntime      = 1u;
constexpr uint32_t kEvInvalidVote   = 2u;

constexpr uint64_t kRewardScale = 1000000000000000000ULL;

// keccak-f[1600]
__constant__ static const uint64_t kKeccakRC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808AULL, 0x8000000080008000ULL,
    0x000000000000808BULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008AULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800AULL, 0x800000008000000AULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL,
};

__constant__ static const uint32_t kKeccakRot[25] = {
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14,
};

// rotl64 is called with n=0 in the rho-pi step (kKeccakRot[0] == 0).
// `x >> (64 - n)` is UB at n=0; mask the shift count to make it well-defined.
__device__ inline uint64_t rotl64(uint64_t x, uint32_t n) {
    return (x << (n & 63u)) | (x >> ((64u - n) & 63u));
}

__device__ inline void keccak_f1600(uint64_t* s) {
    for (uint32_t round = 0; round < 24u; ++round) {
        uint64_t c[5];
        for (uint32_t x = 0; x < 5u; ++x)
            c[x] = s[x] ^ s[x+5] ^ s[x+10] ^ s[x+15] ^ s[x+20];
        uint64_t d[5];
        for (uint32_t x = 0; x < 5u; ++x)
            d[x] = c[(x + 4u) % 5u] ^ rotl64(c[(x + 1u) % 5u], 1u);
        for (uint32_t y = 0; y < 25u; y += 5u)
            for (uint32_t x = 0; x < 5u; ++x)
                s[y + x] ^= d[x];
        uint64_t b[25];
        for (uint32_t y = 0; y < 5u; ++y)
            for (uint32_t x = 0; x < 5u; ++x) {
                uint32_t i = x + 5u * y;
                uint32_t j = y + 5u * ((2u * x + 3u * y) % 5u);
                b[j] = rotl64(s[i], kKeccakRot[i]);
            }
        for (uint32_t y = 0; y < 25u; y += 5u) {
            uint64_t t0 = b[y+0], t1 = b[y+1], t2 = b[y+2], t3 = b[y+3], t4 = b[y+4];
            s[y+0] = t0 ^ ((~t1) & t2);
            s[y+1] = t1 ^ ((~t2) & t3);
            s[y+2] = t2 ^ ((~t3) & t4);
            s[y+3] = t3 ^ ((~t4) & t0);
            s[y+4] = t4 ^ ((~t0) & t1);
        }
        s[0] ^= kKeccakRC[round];
    }
}

__device__ inline void keccak256(const uint8_t* data, uint64_t len, uint8_t* out) {
    uint64_t s[25] = {0};
    constexpr uint32_t rate = 136u;
    uint64_t off = 0;
    while (len - off >= rate) {
        for (uint32_t i = 0; i < rate; ++i) {
            uint32_t lane = i / 8u, sh = (i % 8u) * 8u;
            s[lane] ^= ((uint64_t)data[off + i]) << sh;
        }
        keccak_f1600(s);
        off += rate;
    }
    uint8_t block[136] = {0};
    uint64_t rem = len - off;
    for (uint64_t i = 0; i < rem; ++i) block[i] = data[off + i];
    block[rem]      ^= 0x01;
    block[rate - 1] ^= 0x80;
    for (uint32_t i = 0; i < rate; ++i) {
        uint32_t lane = i / 8u, sh = (i % 8u) * 8u;
        s[lane] ^= ((uint64_t)block[i]) << sh;
    }
    keccak_f1600(s);
    for (uint32_t i = 0; i < 32u; ++i) {
        uint32_t lane = i / 8u, sh = (i % 8u) * 8u;
        out[i] = (uint8_t)((s[lane] >> sh) & 0xFFu);
    }
}

__device__ inline void absorb_u32(uint8_t* dst, uint32_t off, uint32_t v) {
    for (uint32_t k = 0; k < 4u; ++k) dst[off + k] = (uint8_t)((v >> (k*8u)) & 0xFFu);
}

__device__ inline void absorb_u64(uint8_t* dst, uint32_t off, uint64_t v) {
    for (uint32_t k = 0; k < 8u; ++k) dst[off + k] = (uint8_t)((v >> (k*8u)) & 0xFFu);
}

__device__ inline uint64_t sat_add_u64(uint64_t a, uint64_t b) {
    uint64_t r = a + b;
    return (r < a) ? UINT64_MAX : r;
}

__device__ inline uint64_t sat_sub_u64(uint64_t a, uint64_t b) {
    return (a < b) ? 0u : (a - b);
}

__device__ inline uint32_t validator_index_hash(uint64_t validator_id, uint32_t mask) {
    uint64_t h = 0xcbf29ce484222325ULL;
    h = (h ^ validator_id) * 0x100000001b3ULL;
    return (uint32_t)h & mask;
}

__device__ inline uint32_t validator_locate(ValidatorSlot* tab, uint32_t count,
                                            uint64_t validator_id, bool insert_if_missing)
{
    uint32_t mask = count - 1u;
    uint32_t idx  = validator_index_hash(validator_id, mask);
    for (uint32_t probe = 0; probe < count; ++probe) {
        ValidatorSlot& s = tab[idx];
        if (s.occupied == 0u) {
            if (insert_if_missing) {
                s.validator_id = validator_id;
                s.weight = 0;
                s.status = 0;
                s.jail_until_epoch = 0;
                s.occupied = 1u;
                for (uint32_t k = 0; k < 48u; ++k) s.bls_pubkey[k] = 0;
                for (uint32_t k = 0; k < 32u; ++k) s.ringtail_pubkey[k] = 0;
                for (uint32_t k = 0; k < 32u; ++k) s.mldsa_pubkey[k] = 0;
                for (uint32_t k = 0; k < 32u; ++k) s.mldsa_groth16_root[k] = 0;
                return idx;
            }
            return 0xFFFFFFFFu;
        }
        if (s.validator_id == validator_id) return idx;
        idx = (idx + 1u) & mask;
    }
    return 0xFFFFFFFFu;
}

__device__ inline uint32_t stake_record_index_hash(uint64_t delegator,
                                                   uint64_t validator,
                                                   uint32_t mask) {
    uint64_t composite = delegator ^ (validator + 0x9E3779B97F4A7C15ULL +
                                      (delegator << 6) + (delegator >> 2));
    return (uint32_t)composite & mask;
}

__device__ inline uint32_t stake_record_locate(StakeRecord* tab, uint32_t count,
                                               uint64_t delegator, uint64_t validator,
                                               bool insert_if_missing)
{
    uint32_t mask = count - 1u;
    uint32_t idx  = stake_record_index_hash(delegator, validator, mask);
    for (uint32_t probe = 0; probe < count; ++probe) {
        StakeRecord& s = tab[idx];
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

}  // namespace pvm::cuda
