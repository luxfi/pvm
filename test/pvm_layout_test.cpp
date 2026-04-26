// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file pvm_layout_test.cpp
/// PVM v0.52 — layout invariants for cross-backend determinism.
///
/// Validates struct sizes, alignment, member offsets, and basic CPU-reference
/// determinism on a small canonical workload. Anything that drifts in the
/// host header without a parallel update in the GPU kernels would break the
/// CPU/Metal/CUDA equivalence — these checks are the first line of defense.

#include "lux/pvm/pvm_gpu_layout.hpp"
#include "lux/pvm/pvm_cpu_reference.hpp"

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace pvm::gpu;

namespace {

int g_passed = 0;
int g_failed = 0;

#define EXPECT(name, cond)                                                  \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::printf("  FAIL[%s]: %s\n", (name), #cond);                 \
            std::fflush(stdout);                                            \
            ++g_failed;                                                     \
            return;                                                         \
        }                                                                   \
    } while (0)

#define PASS(name)                                                          \
    do {                                                                    \
        std::printf("  ok  : %s\n", (name));                                \
        std::fflush(stdout);                                                \
        ++g_passed;                                                         \
    } while (0)

void test_validator_slot_layout()
{
    // 8 (id) + 8 (weight) + 48 + 32 + 32 + 32 (keys) + 4 + 4 + 4 + 4 = 176
    EXPECT("ValidatorSlot.size", sizeof(ValidatorSlot) == 176);
    EXPECT("ValidatorSlot.align", alignof(ValidatorSlot) == 16);
    EXPECT("ValidatorSlot.id.off",        offsetof(ValidatorSlot, validator_id)        == 0);
    EXPECT("ValidatorSlot.weight.off",    offsetof(ValidatorSlot, weight)              == 8);
    EXPECT("ValidatorSlot.bls.off",       offsetof(ValidatorSlot, bls_pubkey)          == 16);
    EXPECT("ValidatorSlot.ringtail.off",  offsetof(ValidatorSlot, ringtail_pubkey)     == 64);
    EXPECT("ValidatorSlot.mldsa.off",     offsetof(ValidatorSlot, mldsa_pubkey)        == 96);
    EXPECT("ValidatorSlot.groth16.off",   offsetof(ValidatorSlot, mldsa_groth16_root)  == 128);
    EXPECT("ValidatorSlot.status.off",    offsetof(ValidatorSlot, status)              == 160);
    EXPECT("ValidatorSlot.jail.off",      offsetof(ValidatorSlot, jail_until_epoch)    == 164);
    EXPECT("ValidatorSlot.occupied.off",  offsetof(ValidatorSlot, occupied)            == 168);
    PASS("ValidatorSlot layout");
}

void test_stake_record_layout()
{
    // 8 * 5 + 4 * 4 + 8 (pad) = 64
    EXPECT("StakeRecord.size",  sizeof(StakeRecord) == 64);
    EXPECT("StakeRecord.align", alignof(StakeRecord) == 16);
    EXPECT("StakeRecord.deleg.off",       offsetof(StakeRecord, delegator_id)       == 0);
    EXPECT("StakeRecord.val.off",         offsetof(StakeRecord, validator_id)       == 8);
    EXPECT("StakeRecord.amount.off",      offsetof(StakeRecord, amount)             == 16);
    EXPECT("StakeRecord.lock.off",        offsetof(StakeRecord, lock_until_epoch)   == 24);
    EXPECT("StakeRecord.reward.off",      offsetof(StakeRecord, reward_accumulator) == 32);
    EXPECT("StakeRecord.commission.off",  offsetof(StakeRecord, commission_bps)     == 40);
    EXPECT("StakeRecord.status.off",      offsetof(StakeRecord, status)             == 44);
    EXPECT("StakeRecord.epoch_b.off",     offsetof(StakeRecord, epoch_bonded)       == 48);
    EXPECT("StakeRecord.epoch_u.off",     offsetof(StakeRecord, epoch_unbonded)     == 52);
    PASS("StakeRecord layout");
}

void test_stake_op_layout()
{
    // 8 * 5 + 4 * 4 + 8 (pad) = 64
    EXPECT("StakeOp.size",  sizeof(StakeOp) == 64);
    EXPECT("StakeOp.align", alignof(StakeOp) == 16);
    EXPECT("StakeOp.kind.off", offsetof(StakeOp, kind) == 40);
    PASS("StakeOp layout");
}

void test_slash_evidence_layout()
{
    // 8*3 + 4*4 + 32 + 8 (pad) = 80
    EXPECT("SlashEvidence.size",  sizeof(SlashEvidence) == 80);
    EXPECT("SlashEvidence.align", alignof(SlashEvidence) == 16);
    EXPECT("SlashEvidence.id.off",       offsetof(SlashEvidence, validator_id)     == 0);
    EXPECT("SlashEvidence.height.off",   offsetof(SlashEvidence, height)           == 8);
    EXPECT("SlashEvidence.amount.off",   offsetof(SlashEvidence, slash_amount)     == 16);
    EXPECT("SlashEvidence.kind.off",     offsetof(SlashEvidence, kind)             == 24);
    EXPECT("SlashEvidence.epoch.off",    offsetof(SlashEvidence, epoch)            == 28);
    EXPECT("SlashEvidence.jail.off",     offsetof(SlashEvidence, jail_for_epochs)  == 32);
    EXPECT("SlashEvidence.digest.off",   offsetof(SlashEvidence, evidence_digest)  == 40);
    PASS("SlashEvidence layout");
}

void test_epoch_state_layout()
{
    // 8*3 + 4*2 + 32*4 = 24 + 8 + 128 = 160
    EXPECT("EpochState.size",  sizeof(EpochState) == 160);
    EXPECT("EpochState.align", alignof(EpochState) == 16);
    EXPECT("EpochState.curr.off",  offsetof(EpochState, current_epoch)         == 0);
    EXPECT("EpochState.next.off",  offsetof(EpochState, next_epoch_height)     == 8);
    EXPECT("EpochState.total.off", offsetof(EpochState, total_active_stake)    == 16);
    EXPECT("EpochState.actc.off",  offsetof(EpochState, active_validator_count) == 24);
    EXPECT("EpochState.pdrp.off",  offsetof(EpochState, pending_drop_count)    == 28);
    EXPECT("EpochState.vroot.off", offsetof(EpochState, validator_set_root)    == 32);
    EXPECT("EpochState.sroot.off", offsetof(EpochState, stake_root)            == 64);
    EXPECT("EpochState.slroot.off",offsetof(EpochState, slashing_root)         == 96);
    EXPECT("EpochState.eroot.off", offsetof(EpochState, epoch_root)            == 128);
    PASS("EpochState layout");
}

void test_validator_op_layout()
{
    // 8 + 8 + 48 + 32*3 + 4*4 = 16 + 48 + 96 + 16 = 176
    EXPECT("ValidatorOp.size",  sizeof(ValidatorOp) == 176);
    EXPECT("ValidatorOp.align", alignof(ValidatorOp) == 16);
    EXPECT("ValidatorOp.bls.off",     offsetof(ValidatorOp, bls_pubkey)         == 16);
    EXPECT("ValidatorOp.kind.off",    offsetof(ValidatorOp, kind)               == 160);
    PASS("ValidatorOp layout");
}

void test_round_descriptor_layout()
{
    // 8*4 + 4*6 + 8 + 32 = 32 + 24 + 8 + 32 = 96
    EXPECT("PVMRoundDescriptor.size",  sizeof(PVMRoundDescriptor) == 96);
    EXPECT("PVMRoundDescriptor.align", alignof(PVMRoundDescriptor) == 16);
    EXPECT("Desc.mode.off",  offsetof(PVMRoundDescriptor, mode)  == 32);
    EXPECT("Desc.parent.off",offsetof(PVMRoundDescriptor, parent_epoch_root) == 64);
    PASS("PVMRoundDescriptor layout");
}

void test_transition_result_layout()
{
    // 4*8 + 8*4 + 32*4 = 32 + 32 + 128 = 192
    EXPECT("PVMTransitionResult.size",  sizeof(PVMTransitionResult) == 192);
    EXPECT("PVMTransitionResult.align", alignof(PVMTransitionResult) == 16);
    EXPECT("Result.epoch_root.off", offsetof(PVMTransitionResult, epoch_root) == 160);
    PASS("PVMTransitionResult layout");
}

PVMRoundDescriptor make_desc(uint64_t round, uint32_t mode = 4)
{
    PVMRoundDescriptor d{};
    d.chain_id = 1u;
    d.round = round;
    d.timestamp_ns = 1700000000000000000ULL;
    d.epoch = 0u;
    d.mode = mode;       // FullRound default
    d.closing_flag = 1u; // close epoch by default
    return d;
}

ValidatorOp make_validator_add(uint64_t id, uint64_t weight, uint8_t fill = 0xAB)
{
    ValidatorOp op{};
    op.validator_id = id;
    op.weight = weight;
    op.kind = static_cast<uint32_t>(ValidatorOpKind::Add);
    for (auto& b : op.bls_pubkey) b = fill;
    for (auto& b : op.ringtail_pubkey) b = fill ^ 1u;
    for (auto& b : op.mldsa_pubkey) b = fill ^ 2u;
    for (auto& b : op.mldsa_groth16_root) b = fill ^ 3u;
    return op;
}

StakeOp make_bond(uint64_t deleg, uint64_t val, uint64_t amount)
{
    StakeOp op{};
    op.delegator_id = deleg;
    op.validator_id = val;
    op.amount = amount;
    op.kind = static_cast<uint32_t>(StakeOpKind::Bond);
    return op;
}

void test_cpu_reference_determinism()
{
    // Build a small canonical input.
    std::vector<ValidatorOp> v_ops;
    for (uint64_t i = 1; i <= 8; ++i)
        v_ops.push_back(make_validator_add(i, 1000u * i));

    std::vector<StakeOp> s_ops;
    for (uint64_t d = 100; d < 116; ++d)
        s_ops.push_back(make_bond(d, (d % 8u) + 1u, 500u));

    std::vector<SlashEvidence> ev_ops;
    SlashEvidence ev{};
    ev.validator_id = 5u;
    ev.height = 12345u;
    ev.kind = static_cast<uint32_t>(SlashEvidenceKind::Equivocation);
    ev.epoch = 0u;
    for (auto& b : ev.evidence_digest) b = 0xCC;
    ev_ops.push_back(ev);

    auto desc = make_desc(1u, /*mode=*/4u);

    // Run the reference twice and require byte-identical results.
    auto state1 = ref::PVMReferenceState::empty();
    auto state2 = ref::PVMReferenceState::empty();
    auto r1 = ref::run_reference(state1, desc, v_ops, s_ops, ev_ops);
    auto r2 = ref::run_reference(state2, desc, v_ops, s_ops, ev_ops);

    EXPECT("ref.det.status",   r1.status == 1u);
    EXPECT("ref.det.vapply",   r1.validator_apply_count == 8u);
    EXPECT("ref.det.sapply",   r1.stake_apply_count == 16u);
    EXPECT("ref.det.slapply",  r1.slash_apply_count == 1u);

    EXPECT("ref.det.vroot",    std::memcmp(r1.validator_set_root, r2.validator_set_root, 32) == 0);
    EXPECT("ref.det.sroot",    std::memcmp(r1.stake_root,         r2.stake_root,         32) == 0);
    EXPECT("ref.det.slroot",   std::memcmp(r1.slashing_root,      r2.slashing_root,      32) == 0);
    EXPECT("ref.det.eroot",    std::memcmp(r1.epoch_root,         r2.epoch_root,         32) == 0);

    EXPECT("ref.det.epoch",    r1.epoch == 1u);

    // epoch_root must be non-zero (some validator + stake activity).
    bool any_nonzero = false;
    for (auto b : r1.epoch_root) if (b != 0) { any_nonzero = true; break; }
    EXPECT("ref.det.eroot.nz", any_nonzero);

    // Tombstoned validator (5) must NOT count as active.
    EXPECT("ref.det.active",   r1.active_validator_count == 7u);
    EXPECT("ref.det.tomb",     r1.tombstoned_count == 1u);

    PASS("CPU reference determinism");
}

void test_cpu_reference_jail_excludes()
{
    std::vector<ValidatorOp> v_ops;
    v_ops.push_back(make_validator_add(1u, 1000u));
    v_ops.push_back(make_validator_add(2u, 2000u));

    // Jail validator 1 until epoch 10.
    ValidatorOp jail{};
    jail.validator_id = 1u;
    jail.kind = static_cast<uint32_t>(ValidatorOpKind::Jail);
    jail.jail_until_epoch = 10u;
    v_ops.push_back(jail);

    auto desc = make_desc(1u, 4u);
    auto state = ref::PVMReferenceState::empty();
    auto r = ref::run_reference(state, desc, v_ops, {}, {});

    EXPECT("jail.active", r.active_validator_count == 1u);  // only validator 2
    EXPECT("jail.jailed", r.jailed_count == 1u);
    PASS("Jailed validator excluded");
}

void test_cpu_reference_redelegate_atomic()
{
    std::vector<ValidatorOp> v_ops;
    v_ops.push_back(make_validator_add(1u, 0u));
    v_ops.push_back(make_validator_add(2u, 0u));

    std::vector<StakeOp> s_ops;
    // Bond 500 to v1
    s_ops.push_back(make_bond(100u, 1u, 500u));

    // Redelegate 300 from v1 -> v2
    StakeOp rd{};
    rd.delegator_id = 100u;
    rd.validator_id = 2u;            // dest
    rd.source_validator_id = 1u;     // src
    rd.amount = 300u;
    rd.kind = static_cast<uint32_t>(StakeOpKind::Redelegate);
    s_ops.push_back(rd);

    auto desc = make_desc(1u, 4u);
    auto state = ref::PVMReferenceState::empty();
    auto r = ref::run_reference(state, desc, v_ops, s_ops, {});

    // After: v1 weight = 200, v2 weight = 300, total = 500
    EXPECT("rd.active", r.active_validator_count == 2u);
    EXPECT("rd.total",  r.total_active_stake == 500u);
    EXPECT("rd.applied",r.stake_apply_count == 2u);
    PASS("Redelegate atomic");
}

void test_cpu_reference_empty_round()
{
    auto desc = make_desc(1u, 4u);
    auto state = ref::PVMReferenceState::empty();
    auto r = ref::run_reference(state, desc, {}, {}, {});

    EXPECT("empty.status", r.status == 1u);
    EXPECT("empty.epoch",  r.epoch == 1u);

    // epoch_root for empty epoch: still deterministic, but composed over zero arenas.
    auto state2 = ref::PVMReferenceState::empty();
    auto r2 = ref::run_reference(state2, desc, {}, {}, {});
    EXPECT("empty.eroot.det", std::memcmp(r.epoch_root, r2.epoch_root, 32) == 0);

    bool any_nonzero = false;
    for (auto b : r.epoch_root) if (b != 0) { any_nonzero = true; break; }
    EXPECT("empty.eroot.nz", any_nonzero);

    PASS("Empty round deterministic");
}

}  // namespace

int main(int /*argc*/, char** /*argv*/)
{
    setvbuf(stdout, nullptr, _IOLBF, 0);
    std::printf("[pvm_layout_test] starting\n");

    test_validator_slot_layout();
    test_stake_record_layout();
    test_stake_op_layout();
    test_slash_evidence_layout();
    test_epoch_state_layout();
    test_validator_op_layout();
    test_round_descriptor_layout();
    test_transition_result_layout();

    test_cpu_reference_determinism();
    test_cpu_reference_jail_excludes();
    test_cpu_reference_redelegate_atomic();
    test_cpu_reference_empty_round();

    std::printf("[pvm_layout_test] passed=%d failed=%d\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
