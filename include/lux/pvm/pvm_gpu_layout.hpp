// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file pvm_gpu_layout.hpp
/// Shared host/GPU memory layouts for the PVM (Platform-VM) GPU substrate.
///
/// **Scope** — this module is the **GPU-native transition substrate for the
/// P-Chain**. State (validators, stake, slashing) lives on the GPU and the
/// canonical transition logic also runs on the GPU. This is what closes the
/// LP-137 gap from "GPU-resident" to "GPU-native": a small dedicated set of
/// transition kernels that match a deterministic CPU reference byte-for-byte.
///
///   pvm/                           the substrate this header describes
///   pvm/src/pvm_cpu_reference.cpp  deterministic CPU oracle (this file's twin)
///   pvm/src/pvm_*.cu               CUDA transition kernels (v0.53)
///   pvm/src/pvm_*.metal            Metal transition kernels (v0.53)
///
/// Round model — one PVMTransitionRound covers:
///   * ValidatorSetApply  : add/remove/update weight, rotate keys, jail/unjail
///   * StakeTransition    : bond / unbond / delegate / redelegate / reward
///   * SlashingTransition : evidence ingest -> slash + jail update
///   * EpochTransition    : close epoch, compute next validator set,
///                          emit pchain_validator_root for Quasar binding
///
/// All offsets here MUST match pvm_*.metal and pvm_*.cu byte-for-byte. The
/// CPU reference implementation in pvm_cpu_reference.cpp consumes the same
/// arenas in the same order and produces identical roots — it is the
/// equivalence oracle for cross-backend determinism (CPU vs Metal vs CUDA).

#pragma once

#include <cstdint>

namespace pvm::gpu {

// =============================================================================
// Residency class tags
// =============================================================================
//
// PVM splits its working set into two residency classes:
//
//   * DeviceHot  — validator set, active stake, active slashing window,
//                  current epoch state. Pinned in GPU memory; transitions
//                  read and write here.
//   * HostCold   — archival epochs, historical slashing evidence, expired
//                  stake records. Resident on the host; pulled into the
//                  device only on faulted access (out of scope for v0.52).
//
// A transition kernel only reads/writes DeviceHot arenas. Anything that
// would force a cold read promotes the round to a faulting state and is
// punted back to the host scheduler. v0.52 ships the DeviceHot path only.

enum class ResidencyClass : uint32_t {
    DeviceHot = 0,
    HostCold  = 1,
};

// =============================================================================
// Validator arena (DeviceHot)
// =============================================================================
//
// Open-addressing table keyed by validator_id. Each slot holds a full
// committee record: BLS / Ringtail / ML-DSA public-key slots, Z-Chain
// MLDSA->Groth16 root commitment (so the Z-Chain ceremony can prove the
// validator's identity binding), stake weight, status flags, and a
// jail-until counter expressed in epochs.
//
// status bits (additive, OR-ed):
//   0x1  active        — counted in the validator set
//   0x2  jailed        — temporarily excluded; jail_until in epochs
//   0x4  tombstoned    — permanent removal (double-sign / catastrophic)
//   0x8  pending_add   — staged for next epoch's set
//   0x10 pending_drop  — staged for removal at next epoch boundary

struct alignas(16) ValidatorSlot {
    uint64_t validator_id;        ///< 0 means empty (gated by occupied flag)
    uint64_t weight;              ///< stake weight (in nLUX)
    uint8_t  bls_pubkey[48];      ///< BLS12-381 G1 (compressed)
    uint8_t  ringtail_pubkey[32]; ///< Ringtail commitment (digest only)
    uint8_t  mldsa_pubkey[32];    ///< ML-DSA-65 commitment (digest only)
    uint8_t  mldsa_groth16_root[32]; ///< Z-Chain ceremony root binding ML-DSA -> Groth16
    uint32_t status;              ///< status bits above
    uint32_t jail_until_epoch;    ///< epoch at which the validator may rejoin
    uint32_t occupied;            ///< 0=free, 1=occupied
    uint32_t _pad0;
};
static_assert(sizeof(ValidatorSlot) == 8 + 8 + 48 + 32 + 32 + 32 + 16,
              "ValidatorSlot layout drift");
static_assert(alignof(ValidatorSlot) == 16, "ValidatorSlot alignment drift");

inline constexpr uint32_t kDefaultValidatorSlots = 256u;

// =============================================================================
// Stake arena (DeviceHot)
// =============================================================================
//
// Append-only log of bonded stake records. v0.52 models these as discrete
// records keyed by (delegator, validator). v0.53 transitions consume one
// StakeOp at a time and either insert / mutate / retire records here.
//
// reward_accumulator is a fixed-point monotonic counter (1e18 scale) used
// to compute pro-rata reward distribution at epoch close. Overflow is
// caught and saturated; the EpochTransition kernel performs the full
// settle and zeroes the accumulator.

struct alignas(16) StakeRecord {
    uint64_t delegator_id;
    uint64_t validator_id;
    uint64_t amount;              ///< nLUX bonded
    uint64_t lock_until_epoch;    ///< 0 means unlocked
    uint64_t reward_accumulator;  ///< fixed-point 1e18, saturating
    uint32_t commission_bps;      ///< validator commission in basis points (0..10000)
    uint32_t status;              ///< 0=free, 1=active, 2=unbonding, 3=retired
    uint32_t epoch_bonded;
    uint32_t epoch_unbonded;      ///< 0 if still bonded
    uint64_t _pad0;               ///< pad to 64 bytes for arena stride
};
static_assert(sizeof(StakeRecord) == 64, "StakeRecord layout drift");
static_assert(alignof(StakeRecord) == 16, "StakeRecord alignment drift");

inline constexpr uint32_t kDefaultStakeSlots = 4096u;

// Stake operation kinds (host-supplied input to StakeTransition).
enum class StakeOpKind : uint32_t {
    Bond       = 0,
    Unbond     = 1,
    Delegate   = 2,
    Redelegate = 3,
    Reward     = 4,
    Commission = 5,
};

struct alignas(16) StakeOp {
    uint64_t delegator_id;
    uint64_t validator_id;
    uint64_t amount;
    uint64_t lock_until_epoch;
    uint64_t source_validator_id; ///< Redelegate only; 0 otherwise
    uint32_t kind;                ///< StakeOpKind
    uint32_t commission_bps;
    uint32_t epoch;
    uint32_t _pad0;
    uint64_t _pad1;               ///< pad to 64 bytes for arena stride
};
static_assert(sizeof(StakeOp) == 64, "StakeOp layout drift");
static_assert(alignof(StakeOp) == 16, "StakeOp alignment drift");

// =============================================================================
// Slashing arena (DeviceHot)
// =============================================================================

enum class SlashEvidenceKind : uint32_t {
    Equivocation = 0,   ///< double-signing in same height
    Downtime     = 1,   ///< missed > N consecutive proposals
    InvalidVote  = 2,   ///< signed an invalidated subject
};

struct alignas(16) SlashEvidence {
    uint64_t validator_id;
    uint64_t height;              ///< block / round height
    uint64_t slash_amount;        ///< nLUX to slash; if 0, kernel computes from policy
    uint32_t kind;                ///< SlashEvidenceKind
    uint32_t epoch;
    uint32_t jail_for_epochs;     ///< 0 means "use policy default"
    uint32_t _pad0;
    uint8_t  evidence_digest[32]; ///< keccak of the proof blob
    uint64_t _pad1;               ///< pad to 80 bytes for arena stride
};
static_assert(sizeof(SlashEvidence) == 80, "SlashEvidence layout drift");
static_assert(alignof(SlashEvidence) == 16, "SlashEvidence alignment drift");

inline constexpr uint32_t kDefaultSlashSlots = 1024u;

// =============================================================================
// Epoch arena (DeviceHot)
// =============================================================================
//
// One slot per active epoch (the previous epoch's record is rolled into
// the chain's archive at close time). Roots here feed the Quasar round
// descriptor's pchain_validator_root binding.

struct alignas(16) EpochState {
    uint64_t current_epoch;
    uint64_t next_epoch_height;       ///< height at which next epoch activates
    uint64_t total_active_stake;
    uint32_t active_validator_count;
    uint32_t pending_drop_count;
    uint8_t  validator_set_root[32];  ///< keccak over occupied ValidatorSlot keys
    uint8_t  stake_root[32];          ///< keccak over active StakeRecord keys
    uint8_t  slashing_root[32];       ///< keccak over SlashEvidence digests
    uint8_t  epoch_root[32];          ///< composed root (== pchain_validator_root)
};
static_assert(sizeof(EpochState) == 8*3 + 4*2 + 32*4, "EpochState layout drift");
static_assert(alignof(EpochState) == 16, "EpochState alignment drift");

// =============================================================================
// Round descriptor (host -> GPU, written once per round)
// =============================================================================

enum class PVMTransitionMode : uint32_t {
    ValidatorSetApply    = 0,
    StakeTransition      = 1,
    SlashingTransition   = 2,
    EpochTransition      = 3,
    FullRound            = 4,   ///< chain all four in canonical order
};

struct alignas(16) PVMRoundDescriptor {
    uint64_t chain_id;            ///< P-Chain canonical id
    uint64_t round;               ///< monotonic
    uint64_t timestamp_ns;
    uint64_t epoch;               ///< current epoch at round start
    uint32_t mode;                ///< PVMTransitionMode
    uint32_t validator_op_count;
    uint32_t stake_op_count;
    uint32_t slash_evidence_count;
    uint32_t closing_flag;        ///< 1 = run EpochTransition after the others
    uint32_t _pad0;
    uint64_t _pad1;
    uint8_t  parent_epoch_root[32];
};
static_assert(sizeof(PVMRoundDescriptor) == 8*4 + 4*6 + 8 + 32,
              "PVMRoundDescriptor layout drift");
static_assert(alignof(PVMRoundDescriptor) == 16,
              "PVMRoundDescriptor alignment drift");

// Validator op (host-supplied input to ValidatorSetApply).
enum class ValidatorOpKind : uint32_t {
    Add          = 0,
    Remove       = 1,
    UpdateWeight = 2,
    Jail         = 3,
    Unjail       = 4,
    RotateKeys   = 5,
};

struct alignas(16) ValidatorOp {
    uint64_t validator_id;
    uint64_t weight;
    uint8_t  bls_pubkey[48];
    uint8_t  ringtail_pubkey[32];
    uint8_t  mldsa_pubkey[32];
    uint8_t  mldsa_groth16_root[32];
    uint32_t kind;                ///< ValidatorOpKind
    uint32_t jail_until_epoch;
    uint32_t epoch;
    uint32_t _pad0;
};
static_assert(sizeof(ValidatorOp) == 8 + 8 + 48 + 32*3 + 4*4,
              "ValidatorOp layout drift");
static_assert(alignof(ValidatorOp) == 16, "ValidatorOp alignment drift");

// =============================================================================
// Round result (GPU -> host)
// =============================================================================

struct alignas(16) PVMTransitionResult {
    uint32_t status;                  ///< 0=in-progress, 1=finalized, 2=needs_state, 3=failed
    uint32_t validator_apply_count;   ///< validator ops successfully applied
    uint32_t stake_apply_count;
    uint32_t slash_apply_count;
    uint32_t active_validator_count;
    uint32_t pending_drop_count;
    uint32_t jailed_count;
    uint32_t tombstoned_count;
    uint64_t total_active_stake;
    uint64_t total_slashed;
    uint64_t total_rewards;
    uint64_t epoch;                   ///< epoch after EpochTransition (== input epoch + 1
                                      ///< on FullRound, otherwise unchanged)
    uint8_t  validator_set_root[32];
    uint8_t  stake_root[32];
    uint8_t  slashing_root[32];
    uint8_t  epoch_root[32];          ///< == pchain_validator_root for Quasar binding
};
static_assert(sizeof(PVMTransitionResult) == 4*8 + 8*4 + 32*4,
              "PVMTransitionResult layout drift");
static_assert(alignof(PVMTransitionResult) == 16,
              "PVMTransitionResult alignment drift");

}  // namespace pvm::gpu
