// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file pvm_cpu_reference.cpp
/// PVM CPU reference — deterministic oracle for cross-backend determinism.
///
/// Mirrors what pvm_*.metal / pvm_*.cu must produce byte-for-byte:
///   * ValidatorSetApply  : table mutations + status transitions
///   * StakeTransition    : per-record bond / unbond / delegate / redelegate
///                          / reward / commission accounting
///   * SlashingTransition : evidence -> stake slash + jail update
///   * EpochTransition    : root computation; epoch_root composes the
///                          three component roots in canonical order
///
/// Determinism contract: identical input -> identical
///   (validators[], stake[], slashing[], epoch) -> identical roots.
/// Where an op references a missing slot or violates an invariant, the
/// reference skips the op silently and does NOT mutate state — same as
/// the GPU kernel will. The xxx_apply_count counters in the result
/// reflect the number of successfully-applied ops, providing a
/// determinism check without surfacing GPU error states.

#include "lux/pvm/pvm_cpu_reference.hpp"

#include <algorithm>
#include <array>
#include <cstring>

namespace pvm::gpu::ref {

namespace {

// =============================================================================
// keccak256 (matches the implementation in cevm/quasar — bit-identical)
// =============================================================================

constexpr std::array<uint64_t, 24> kKeccakRC = {
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

constexpr std::array<uint32_t, 25> kKeccakRot = {
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14,
};

inline uint64_t rotl64(uint64_t x, uint32_t n) {
    return (x << n) | (x >> (64u - n));
}

void keccak_f1600(uint64_t* s) {
    for (uint32_t round = 0; round < 24u; ++round) {
        uint64_t c[5];
        for (uint32_t x = 0; x < 5u; ++x)
            c[x] = s[x] ^ s[x+5] ^ s[x+10] ^ s[x+15] ^ s[x+20];
        uint64_t d[5];
        for (uint32_t x = 0; x < 5u; ++x)
            d[x] = c[(x + 4u) % 5u] ^ rotl64(c[(x + 1u) % 5u], 1);
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

void keccak256(const uint8_t* data, uint64_t len, uint8_t* out) {
    uint64_t s[25] = {};
    constexpr uint32_t rate = 136;
    uint64_t off = 0;
    while (len - off >= rate) {
        for (uint32_t i = 0; i < rate; ++i) {
            uint32_t lane = i / 8u, sh = (i % 8u) * 8u;
            s[lane] ^= uint64_t(data[off + i]) << sh;
        }
        keccak_f1600(s);
        off += rate;
    }
    uint8_t block[rate] = {};
    uint64_t rem = len - off;
    for (uint64_t i = 0; i < rem; ++i) block[i] = data[off + i];
    block[rem]      ^= 0x01;
    block[rate - 1] ^= 0x80;
    for (uint32_t i = 0; i < rate; ++i) {
        uint32_t lane = i / 8u, sh = (i % 8u) * 8u;
        s[lane] ^= uint64_t(block[i]) << sh;
    }
    keccak_f1600(s);
    for (uint32_t i = 0; i < 32u; ++i) {
        uint32_t lane = i / 8u, sh = (i % 8u) * 8u;
        out[i] = uint8_t((s[lane] >> sh) & 0xFFu);
    }
}

void absorb_u32(uint8_t* dst, uint32_t off, uint32_t v) {
    for (uint32_t k = 0; k < 4u; ++k) dst[off + k] = uint8_t((v >> (k*8)) & 0xFFu);
}
void absorb_u64(uint8_t* dst, uint32_t off, uint64_t v) {
    for (uint32_t k = 0; k < 8u; ++k) dst[off + k] = uint8_t((v >> (k*8)) & 0xFFu);
}

// =============================================================================
// Validator-table helpers (open-addressing hash, matches GPU layout)
// =============================================================================

constexpr uint32_t kStatusActive      = 0x1u;
constexpr uint32_t kStatusJailed      = 0x2u;
constexpr uint32_t kStatusTombstoned  = 0x4u;
constexpr uint32_t kStatusPendingAdd  = 0x8u;
constexpr uint32_t kStatusPendingDrop = 0x10u;

uint32_t validator_index(uint64_t validator_id, uint32_t mask) {
    uint64_t h = 0xcbf29ce484222325ULL;
    h = (h ^ validator_id) * 0x100000001b3ULL;
    return uint32_t(h) & mask;
}

uint32_t validator_locate(std::vector<ValidatorSlot>& tab, uint64_t validator_id,
                          bool insert_if_missing)
{
    uint32_t mask = uint32_t(tab.size()) - 1u;
    uint32_t idx  = validator_index(validator_id, mask);
    for (uint32_t probe = 0; probe < tab.size(); ++probe) {
        auto& s = tab[idx];
        if (s.occupied == 0u) {
            if (insert_if_missing) {
                std::memset(&s, 0, sizeof(s));
                s.validator_id = validator_id;
                s.occupied = 1u;
                return idx;
            }
            return 0xFFFFFFFFu;
        }
        if (s.validator_id == validator_id) return idx;
        idx = (idx + 1u) & mask;
    }
    return 0xFFFFFFFFu;
}

uint32_t stake_record_locate(std::vector<StakeRecord>& tab,
                             uint64_t delegator, uint64_t validator,
                             bool insert_if_missing)
{
    uint32_t mask = uint32_t(tab.size()) - 1u;
    uint64_t composite = delegator ^ (validator + 0x9E3779B97F4A7C15ULL +
                                      (delegator << 6) + (delegator >> 2));
    uint32_t idx = uint32_t(composite) & mask;
    for (uint32_t probe = 0; probe < tab.size(); ++probe) {
        auto& s = tab[idx];
        if (s.status == 0u) {
            if (insert_if_missing) {
                std::memset(&s, 0, sizeof(s));
                s.delegator_id = delegator;
                s.validator_id = validator;
                s.status = 1u;
                return idx;
            }
            return 0xFFFFFFFFu;
        }
        if (s.delegator_id == delegator && s.validator_id == validator) return idx;
        idx = (idx + 1u) & mask;
    }
    return 0xFFFFFFFFu;
}

// =============================================================================
// Kernel 1: ValidatorSetApply
// =============================================================================

uint32_t apply_validator_ops(PVMReferenceState& state,
                             std::span<const ValidatorOp> ops)
{
    uint32_t applied = 0;
    for (const auto& op : ops) {
        switch (static_cast<ValidatorOpKind>(op.kind)) {
            case ValidatorOpKind::Add: {
                uint32_t idx = validator_locate(state.validators, op.validator_id, true);
                if (idx == 0xFFFFFFFFu) break;
                auto& s = state.validators[idx];
                s.weight = op.weight;
                std::memcpy(s.bls_pubkey, op.bls_pubkey, 48);
                std::memcpy(s.ringtail_pubkey, op.ringtail_pubkey, 32);
                std::memcpy(s.mldsa_pubkey, op.mldsa_pubkey, 32);
                std::memcpy(s.mldsa_groth16_root, op.mldsa_groth16_root, 32);
                s.status = kStatusActive | kStatusPendingAdd;
                s.jail_until_epoch = 0;
                ++applied;
                break;
            }
            case ValidatorOpKind::Remove: {
                uint32_t idx = validator_locate(state.validators, op.validator_id, false);
                if (idx == 0xFFFFFFFFu) break;
                auto& s = state.validators[idx];
                if ((s.status & kStatusTombstoned) != 0u) break;
                s.status |= kStatusPendingDrop;
                s.status &= ~kStatusActive;
                ++applied;
                break;
            }
            case ValidatorOpKind::UpdateWeight: {
                uint32_t idx = validator_locate(state.validators, op.validator_id, false);
                if (idx == 0xFFFFFFFFu) break;
                auto& s = state.validators[idx];
                if ((s.status & kStatusTombstoned) != 0u) break;
                s.weight = op.weight;
                ++applied;
                break;
            }
            case ValidatorOpKind::Jail: {
                uint32_t idx = validator_locate(state.validators, op.validator_id, false);
                if (idx == 0xFFFFFFFFu) break;
                auto& s = state.validators[idx];
                if ((s.status & kStatusTombstoned) != 0u) break;
                s.status |= kStatusJailed;
                s.status &= ~kStatusActive;
                if (op.jail_until_epoch > s.jail_until_epoch)
                    s.jail_until_epoch = op.jail_until_epoch;
                ++applied;
                break;
            }
            case ValidatorOpKind::Unjail: {
                uint32_t idx = validator_locate(state.validators, op.validator_id, false);
                if (idx == 0xFFFFFFFFu) break;
                auto& s = state.validators[idx];
                if ((s.status & kStatusTombstoned) != 0u) break;
                if (op.epoch < s.jail_until_epoch) break;
                s.status &= ~kStatusJailed;
                s.status |= kStatusActive;
                s.jail_until_epoch = 0;
                ++applied;
                break;
            }
            case ValidatorOpKind::RotateKeys: {
                uint32_t idx = validator_locate(state.validators, op.validator_id, false);
                if (idx == 0xFFFFFFFFu) break;
                auto& s = state.validators[idx];
                if ((s.status & kStatusTombstoned) != 0u) break;
                std::memcpy(s.bls_pubkey, op.bls_pubkey, 48);
                std::memcpy(s.ringtail_pubkey, op.ringtail_pubkey, 32);
                std::memcpy(s.mldsa_pubkey, op.mldsa_pubkey, 32);
                std::memcpy(s.mldsa_groth16_root, op.mldsa_groth16_root, 32);
                ++applied;
                break;
            }
        }
    }
    return applied;
}

// =============================================================================
// Kernel 2: StakeTransition
// =============================================================================

constexpr uint32_t kStakeStatusActive    = 1u;
constexpr uint32_t kStakeStatusUnbonding = 2u;
constexpr uint32_t kStakeStatusRetired   = 3u;
constexpr uint64_t kRewardScale          = 1'000'000'000'000'000'000ULL; // 1e18

uint64_t saturating_add(uint64_t a, uint64_t b) {
    uint64_t r = a + b;
    return (r < a) ? UINT64_MAX : r;
}

uint64_t saturating_sub(uint64_t a, uint64_t b) {
    return (a < b) ? 0u : (a - b);
}

uint32_t apply_stake_ops(PVMReferenceState& state,
                         std::span<const StakeOp> ops)
{
    uint32_t applied = 0;
    for (const auto& op : ops) {
        switch (static_cast<StakeOpKind>(op.kind)) {
            case StakeOpKind::Bond: {
                uint32_t v_idx = validator_locate(state.validators, op.validator_id, false);
                if (v_idx == 0xFFFFFFFFu) break;
                auto& v = state.validators[v_idx];
                if ((v.status & kStatusTombstoned) != 0u) break;

                uint32_t s_idx = stake_record_locate(state.stake, op.delegator_id,
                                                    op.validator_id, true);
                if (s_idx == 0xFFFFFFFFu) break;
                auto& s = state.stake[s_idx];
                s.amount = saturating_add(s.amount, op.amount);
                if (op.lock_until_epoch > s.lock_until_epoch)
                    s.lock_until_epoch = op.lock_until_epoch;
                if (s.epoch_bonded == 0) s.epoch_bonded = op.epoch;
                s.status = kStakeStatusActive;
                v.weight = saturating_add(v.weight, op.amount);
                ++applied;
                break;
            }
            case StakeOpKind::Unbond: {
                uint32_t s_idx = stake_record_locate(state.stake, op.delegator_id,
                                                    op.validator_id, false);
                if (s_idx == 0xFFFFFFFFu) break;
                auto& s = state.stake[s_idx];
                if (s.status != kStakeStatusActive) break;
                if (op.epoch < s.lock_until_epoch) break;
                uint64_t amt = std::min(op.amount, s.amount);
                s.amount = saturating_sub(s.amount, amt);
                s.epoch_unbonded = op.epoch;
                if (s.amount == 0) s.status = kStakeStatusRetired;
                else s.status = kStakeStatusUnbonding;

                uint32_t v_idx = validator_locate(state.validators, op.validator_id, false);
                if (v_idx != 0xFFFFFFFFu) {
                    auto& v = state.validators[v_idx];
                    v.weight = saturating_sub(v.weight, amt);
                }
                ++applied;
                break;
            }
            case StakeOpKind::Delegate: {
                // Delegate is structurally Bond with explicit delegator semantics.
                uint32_t v_idx = validator_locate(state.validators, op.validator_id, false);
                if (v_idx == 0xFFFFFFFFu) break;
                auto& v = state.validators[v_idx];
                if ((v.status & kStatusTombstoned) != 0u) break;
                if ((v.status & kStatusJailed) != 0u) break;

                uint32_t s_idx = stake_record_locate(state.stake, op.delegator_id,
                                                    op.validator_id, true);
                if (s_idx == 0xFFFFFFFFu) break;
                auto& s = state.stake[s_idx];
                s.amount = saturating_add(s.amount, op.amount);
                s.status = kStakeStatusActive;
                if (s.epoch_bonded == 0) s.epoch_bonded = op.epoch;
                v.weight = saturating_add(v.weight, op.amount);
                ++applied;
                break;
            }
            case StakeOpKind::Redelegate: {
                // Atomic: drain source record, push into destination.
                if (op.source_validator_id == op.validator_id) break;

                uint32_t src_idx = stake_record_locate(state.stake, op.delegator_id,
                                                      op.source_validator_id, false);
                if (src_idx == 0xFFFFFFFFu) break;
                auto& src = state.stake[src_idx];
                if (src.status != kStakeStatusActive) break;
                if (op.epoch < src.lock_until_epoch) break;

                uint32_t v_dst_idx = validator_locate(state.validators, op.validator_id, false);
                if (v_dst_idx == 0xFFFFFFFFu) break;
                auto& v_dst = state.validators[v_dst_idx];
                if ((v_dst.status & kStatusTombstoned) != 0u) break;

                uint64_t amt = std::min(op.amount, src.amount);
                src.amount = saturating_sub(src.amount, amt);
                if (src.amount == 0) src.status = kStakeStatusRetired;

                uint32_t v_src_idx = validator_locate(state.validators,
                                                     op.source_validator_id, false);
                if (v_src_idx != 0xFFFFFFFFu) {
                    auto& v_src = state.validators[v_src_idx];
                    v_src.weight = saturating_sub(v_src.weight, amt);
                }

                uint32_t dst_idx = stake_record_locate(state.stake, op.delegator_id,
                                                      op.validator_id, true);
                if (dst_idx == 0xFFFFFFFFu) break;
                auto& dst = state.stake[dst_idx];
                dst.amount = saturating_add(dst.amount, amt);
                dst.status = kStakeStatusActive;
                if (dst.epoch_bonded == 0) dst.epoch_bonded = op.epoch;
                v_dst.weight = saturating_add(v_dst.weight, amt);
                ++applied;
                break;
            }
            case StakeOpKind::Reward: {
                // Reward credit = op.amount; accumulator advance by amount * 1e18 / weight.
                // Saturating in fixed point. The rewarded weight is the validator's
                // current weight at the time of credit.
                uint32_t v_idx = validator_locate(state.validators, op.validator_id, false);
                if (v_idx == 0xFFFFFFFFu) break;
                auto& v = state.validators[v_idx];
                if (v.weight == 0) break;
                uint64_t scaled = (op.amount > UINT64_MAX / kRewardScale)
                    ? UINT64_MAX
                    : op.amount * kRewardScale;
                uint64_t per_unit = scaled / v.weight;

                // Walk all stake records bonded to this validator, credit them.
                for (auto& s : state.stake) {
                    if (s.status != kStakeStatusActive) continue;
                    if (s.validator_id != op.validator_id) continue;
                    uint64_t delta = (s.amount > UINT64_MAX / per_unit && per_unit != 0)
                        ? UINT64_MAX
                        : s.amount * per_unit;
                    s.reward_accumulator = saturating_add(s.reward_accumulator, delta);
                }
                ++applied;
                break;
            }
            case StakeOpKind::Commission: {
                uint32_t v_idx = validator_locate(state.validators, op.validator_id, false);
                if (v_idx == 0xFFFFFFFFu) break;
                if (op.commission_bps > 10000u) break;
                // commission applies to records owned by validator's self-stake.
                uint32_t s_idx = stake_record_locate(state.stake, op.validator_id,
                                                    op.validator_id, false);
                if (s_idx == 0xFFFFFFFFu) break;
                state.stake[s_idx].commission_bps = op.commission_bps;
                ++applied;
                break;
            }
        }
    }
    return applied;
}

// =============================================================================
// Kernel 3: SlashingTransition
// =============================================================================

uint32_t apply_slash_evidence(PVMReferenceState& state,
                              std::span<const SlashEvidence> evidence,
                              uint64_t& total_slashed)
{
    uint32_t applied = 0;
    total_slashed = 0;
    uint32_t cursor = 0;
    for (const auto& ev : evidence) {
        uint32_t v_idx = validator_locate(state.validators, ev.validator_id, false);
        if (v_idx == 0xFFFFFFFFu) continue;
        auto& v = state.validators[v_idx];
        if ((v.status & kStatusTombstoned) != 0u) continue;

        // Compute slash amount. If host supplied 0, use a policy-based default
        // that is identical to what the GPU kernel will compute.
        uint64_t amount = ev.slash_amount;
        if (amount == 0u) {
            switch (static_cast<SlashEvidenceKind>(ev.kind)) {
                case SlashEvidenceKind::Equivocation: amount = v.weight / 20u; break; //  5%
                case SlashEvidenceKind::Downtime:     amount = v.weight / 100u; break; // 1%
                case SlashEvidenceKind::InvalidVote:  amount = v.weight / 50u; break; // 2%
            }
        }
        if (amount > v.weight) amount = v.weight;
        v.weight = saturating_sub(v.weight, amount);
        total_slashed = saturating_add(total_slashed, amount);

        // Jail / tombstone update.
        if (ev.kind == static_cast<uint32_t>(SlashEvidenceKind::Equivocation)) {
            v.status |= kStatusTombstoned;
            v.status &= ~kStatusActive;
        } else {
            v.status |= kStatusJailed;
            v.status &= ~kStatusActive;
            uint32_t jail_for = ev.jail_for_epochs == 0 ? 100u : ev.jail_for_epochs;
            uint32_t until = ev.epoch + jail_for;
            if (until > v.jail_until_epoch) v.jail_until_epoch = until;
        }

        // Append evidence to slashing arena (ring buffer style).
        if (cursor < state.slashing.size()) {
            state.slashing[cursor] = ev;
            ++cursor;
        }
        ++applied;
    }
    return applied;
}

// =============================================================================
// Kernel 4: EpochTransition (root computation)
// =============================================================================

void compute_validator_set_root(const std::vector<ValidatorSlot>& validators,
                                uint8_t out[32],
                                uint32_t& active_count,
                                uint32_t& jailed_count,
                                uint32_t& tombstoned_count,
                                uint64_t& total_active_stake)
{
    std::array<uint8_t, 32> acc{};
    active_count = jailed_count = tombstoned_count = 0;
    total_active_stake = 0;
    for (uint32_t i = 0; i < validators.size(); ++i) {
        const auto& s = validators[i];
        if (s.occupied == 0u) continue;
        if ((s.status & kStatusTombstoned) != 0u) ++tombstoned_count;
        if ((s.status & kStatusJailed) != 0u) ++jailed_count;
        if ((s.status & kStatusActive) != 0u) {
            ++active_count;
            total_active_stake = saturating_add(total_active_stake, s.weight);
        }

        // leaf = keccak(validator_id || weight || status || jail_until || pubkeys || groth16_root || index)
        uint8_t leaf[8 + 8 + 4 + 4 + 48 + 32 + 32 + 32 + 4] = {};
        uint32_t o = 0;
        absorb_u64(leaf, o, s.validator_id); o += 8;
        absorb_u64(leaf, o, s.weight);       o += 8;
        absorb_u32(leaf, o, s.status);       o += 4;
        absorb_u32(leaf, o, s.jail_until_epoch); o += 4;
        std::memcpy(leaf + o, s.bls_pubkey, 48);          o += 48;
        std::memcpy(leaf + o, s.ringtail_pubkey, 32);     o += 32;
        std::memcpy(leaf + o, s.mldsa_pubkey, 32);        o += 32;
        std::memcpy(leaf + o, s.mldsa_groth16_root, 32);  o += 32;
        absorb_u32(leaf, o, i);              o += 4;

        uint8_t leaf_hash[32];
        keccak256(leaf, o, leaf_hash);

        uint8_t buf[64];
        std::memcpy(buf, acc.data(), 32);
        std::memcpy(buf + 32, leaf_hash, 32);
        keccak256(buf, 64, acc.data());
    }
    std::memcpy(out, acc.data(), 32);
}

void compute_stake_root(const std::vector<StakeRecord>& stake,
                        uint8_t out[32], uint64_t& total_rewards)
{
    std::array<uint8_t, 32> acc{};
    total_rewards = 0;
    for (uint32_t i = 0; i < stake.size(); ++i) {
        const auto& s = stake[i];
        if (s.status == 0u) continue;
        total_rewards = saturating_add(total_rewards, s.reward_accumulator);

        uint8_t leaf[8 + 8 + 8 + 8 + 8 + 4 + 4 + 4 + 4 + 4] = {};
        uint32_t o = 0;
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

        uint8_t leaf_hash[32];
        keccak256(leaf, o, leaf_hash);

        uint8_t buf[64];
        std::memcpy(buf, acc.data(), 32);
        std::memcpy(buf + 32, leaf_hash, 32);
        keccak256(buf, 64, acc.data());
    }
    std::memcpy(out, acc.data(), 32);
}

void compute_slashing_root(const std::vector<SlashEvidence>& slashing,
                           uint8_t out[32])
{
    std::array<uint8_t, 32> acc{};
    for (uint32_t i = 0; i < slashing.size(); ++i) {
        const auto& ev = slashing[i];
        // empty slot if validator_id == 0 AND digest is zero
        bool zero_digest = true;
        for (auto b : ev.evidence_digest) if (b != 0) { zero_digest = false; break; }
        if (ev.validator_id == 0 && zero_digest && ev.height == 0) continue;

        uint8_t leaf[8 + 8 + 8 + 4 + 4 + 4 + 32 + 4] = {};
        uint32_t o = 0;
        absorb_u64(leaf, o, ev.validator_id);      o += 8;
        absorb_u64(leaf, o, ev.height);            o += 8;
        absorb_u64(leaf, o, ev.slash_amount);      o += 8;
        absorb_u32(leaf, o, ev.kind);              o += 4;
        absorb_u32(leaf, o, ev.epoch);             o += 4;
        absorb_u32(leaf, o, ev.jail_for_epochs);   o += 4;
        std::memcpy(leaf + o, ev.evidence_digest, 32); o += 32;
        absorb_u32(leaf, o, i);                    o += 4;

        uint8_t leaf_hash[32];
        keccak256(leaf, o, leaf_hash);

        uint8_t buf[64];
        std::memcpy(buf, acc.data(), 32);
        std::memcpy(buf + 32, leaf_hash, 32);
        keccak256(buf, 64, acc.data());
    }
    std::memcpy(out, acc.data(), 32);
}

void close_epoch(PVMReferenceState& state,
                 const PVMRoundDescriptor& desc,
                 PVMTransitionResult& r)
{
    // Promote pending_add -> active, drop pending_drop -> tombstoned.
    // Auto-unjail validators whose jail_until_epoch has passed.
    uint32_t pending_drop_count = 0;
    uint64_t target_epoch = desc.closing_flag != 0u ? desc.epoch + 1u : desc.epoch;
    for (auto& s : state.validators) {
        if (s.occupied == 0u) continue;
        if ((s.status & kStatusPendingAdd) != 0u) {
            s.status &= ~kStatusPendingAdd;
            // pending_add already implies active per ValidatorOpKind::Add
        }
        if ((s.status & kStatusPendingDrop) != 0u) {
            s.status &= ~kStatusPendingDrop;
            s.status |= kStatusTombstoned;
            ++pending_drop_count;
        }
        if ((s.status & kStatusJailed) != 0u
            && s.jail_until_epoch != 0u
            && uint32_t(target_epoch) >= s.jail_until_epoch
            && (s.status & kStatusTombstoned) == 0u) {
            s.status &= ~kStatusJailed;
            s.status |= kStatusActive;
            s.jail_until_epoch = 0;
        }
    }
    r.pending_drop_count = pending_drop_count;
    state.epoch.pending_drop_count = pending_drop_count;

    // Compute roots (always; even if not closing the epoch we want fresh roots
    // reflecting any mid-round state changes).
    uint32_t active_count = 0, jailed_count = 0, tombstoned_count = 0;
    uint64_t total_active_stake = 0;
    compute_validator_set_root(state.validators,
                               state.epoch.validator_set_root,
                               active_count, jailed_count, tombstoned_count,
                               total_active_stake);
    compute_stake_root(state.stake, state.epoch.stake_root, r.total_rewards);
    compute_slashing_root(state.slashing, state.epoch.slashing_root);

    state.epoch.active_validator_count = active_count;
    state.epoch.total_active_stake = total_active_stake;
    if (desc.closing_flag != 0u) {
        state.epoch.current_epoch = target_epoch;
    }

    // Compose epoch_root = keccak(parent_epoch_root || vset || stake || slashing
    //                            || epoch_u64 || total_stake_u64 || active_count_u32)
    uint8_t composed[32 + 32 + 32 + 32 + 8 + 8 + 4] = {};
    uint32_t o = 0;
    std::memcpy(composed + o, desc.parent_epoch_root, 32);    o += 32;
    std::memcpy(composed + o, state.epoch.validator_set_root, 32); o += 32;
    std::memcpy(composed + o, state.epoch.stake_root, 32);    o += 32;
    std::memcpy(composed + o, state.epoch.slashing_root, 32); o += 32;
    absorb_u64(composed, o, state.epoch.current_epoch);  o += 8;
    absorb_u64(composed, o, state.epoch.total_active_stake); o += 8;
    absorb_u32(composed, o, state.epoch.active_validator_count); o += 4;
    keccak256(composed, o, state.epoch.epoch_root);

    std::memcpy(r.validator_set_root, state.epoch.validator_set_root, 32);
    std::memcpy(r.stake_root,         state.epoch.stake_root,         32);
    std::memcpy(r.slashing_root,      state.epoch.slashing_root,      32);
    std::memcpy(r.epoch_root,         state.epoch.epoch_root,         32);
    r.active_validator_count = active_count;
    r.jailed_count = jailed_count;
    r.tombstoned_count = tombstoned_count;
    r.total_active_stake = total_active_stake;
    r.epoch = state.epoch.current_epoch;
}

}  // anonymous namespace

PVMReferenceState PVMReferenceState::empty() {
    PVMReferenceState s;
    s.validators.assign(kDefaultValidatorSlots, ValidatorSlot{});
    s.stake.assign(kDefaultStakeSlots, StakeRecord{});
    s.slashing.assign(kDefaultSlashSlots, SlashEvidence{});
    s.epoch = EpochState{};
    return s;
}

PVMTransitionResult run_reference(PVMReferenceState& state,
                                  const PVMRoundDescriptor& desc,
                                  std::span<const ValidatorOp>   validator_ops,
                                  std::span<const StakeOp>       stake_ops,
                                  std::span<const SlashEvidence> slash_evidence)
{
    // Lazy-init arenas if caller passed an empty state.
    if (state.validators.empty()) state.validators.assign(kDefaultValidatorSlots, ValidatorSlot{});
    if (state.stake.empty())      state.stake.assign(kDefaultStakeSlots, StakeRecord{});
    if (state.slashing.empty())   state.slashing.assign(kDefaultSlashSlots, SlashEvidence{});

    PVMTransitionResult r{};
    auto mode = static_cast<PVMTransitionMode>(desc.mode);

    if (mode == PVMTransitionMode::ValidatorSetApply ||
        mode == PVMTransitionMode::FullRound) {
        r.validator_apply_count = apply_validator_ops(state, validator_ops);
    }
    if (mode == PVMTransitionMode::StakeTransition ||
        mode == PVMTransitionMode::FullRound) {
        r.stake_apply_count = apply_stake_ops(state, stake_ops);
    }
    if (mode == PVMTransitionMode::SlashingTransition ||
        mode == PVMTransitionMode::FullRound) {
        r.slash_apply_count = apply_slash_evidence(state, slash_evidence,
                                                   r.total_slashed);
    }
    // EpochTransition runs unconditionally so caller always gets fresh roots.
    close_epoch(state, desc, r);
    r.status = 1u;
    return r;
}

}  // namespace pvm::gpu::ref
