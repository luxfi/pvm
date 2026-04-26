// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// pvm_gpu_engine_test.mm — Metal-side correctness for PVMGPUEngine.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "lux/pvm/pvm_gpu_engine.hpp"
#include "lux/pvm/pvm_cpu_reference.hpp"

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

PVMRoundDescriptor make_desc(uint64_t round, uint64_t epoch = 0)
{
    PVMRoundDescriptor d{};
    d.chain_id = 1u;
    d.round = round;
    d.timestamp_ns = 1700000000000000000ULL;
    d.epoch = epoch;
    d.mode = static_cast<uint32_t>(PVMTransitionMode::FullRound);
    d.closing_flag = 1u;
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

void test_engine_creates()
{
    auto engine = PVMGPUEngine::create();
    EXPECT("engine.create", engine != nullptr);
    std::printf("  engine.device: %s\n", engine->device_name());
    PASS("engine creates");
}

void test_engine_full_round_matches_cpu()
{
    auto engine = PVMGPUEngine::create();
    EXPECT("matches.engine", engine != nullptr);

    std::vector<ValidatorOp> v_ops;
    for (uint64_t i = 1; i <= 100; ++i)
        v_ops.push_back(make_validator_add(i, 10000u + i * 7u));

    std::vector<StakeOp> s_ops;
    for (uint64_t d = 1000; d < 2000; ++d)
        s_ops.push_back(make_bond(d, ((d - 1000u) % 100u) + 1u, 100u + (d % 17u)));

    std::vector<SlashEvidence> ev_ops;
    for (uint32_t i = 0; i < 5; ++i) {
        SlashEvidence ev{};
        ev.validator_id = (i + 1u) * 11u;     // 11, 22, 33, 44, 55
        ev.height = 100u * (i + 1u);
        ev.kind = (i % 3u);
        ev.epoch = 0u;
        for (auto& b : ev.evidence_digest) b = uint8_t(0xC0 ^ i);
        ev_ops.push_back(ev);
    }

    auto desc = make_desc(1u);

    // GPU run.
    auto h = engine->begin_round(desc);
    EXPECT("matches.handle", h.valid());
    engine->push_validator_ops(h, v_ops);
    engine->push_stake_ops(h, s_ops);
    engine->push_slash_evidence(h, ev_ops);
    auto gpu_r = engine->run_until_done(h);
    engine->end_round(h);

    // CPU reference.
    auto state = ref::PVMReferenceState::empty();
    auto cpu_r = ref::run_reference(state, desc, v_ops, s_ops, ev_ops);

    EXPECT("matches.status",      gpu_r.status == cpu_r.status);
    EXPECT("matches.vapply",      gpu_r.validator_apply_count == cpu_r.validator_apply_count);
    EXPECT("matches.sapply",      gpu_r.stake_apply_count == cpu_r.stake_apply_count);
    EXPECT("matches.slapply",     gpu_r.slash_apply_count == cpu_r.slash_apply_count);
    EXPECT("matches.active",      gpu_r.active_validator_count == cpu_r.active_validator_count);
    EXPECT("matches.tomb",        gpu_r.tombstoned_count == cpu_r.tombstoned_count);
    EXPECT("matches.jailed",      gpu_r.jailed_count == cpu_r.jailed_count);
    EXPECT("matches.totstake",    gpu_r.total_active_stake == cpu_r.total_active_stake);
    EXPECT("matches.totslash",    gpu_r.total_slashed == cpu_r.total_slashed);
    EXPECT("matches.epoch",       gpu_r.epoch == cpu_r.epoch);

    EXPECT("matches.vroot",       std::memcmp(gpu_r.validator_set_root, cpu_r.validator_set_root, 32) == 0);
    EXPECT("matches.sroot",       std::memcmp(gpu_r.stake_root,         cpu_r.stake_root,         32) == 0);
    EXPECT("matches.slroot",      std::memcmp(gpu_r.slashing_root,      cpu_r.slashing_root,      32) == 0);
    EXPECT("matches.eroot",       std::memcmp(gpu_r.epoch_root,         cpu_r.epoch_root,         32) == 0);

    std::printf("  gpu  vapply=%u sapply=%u slapply=%u active=%u tomb=%u jail=%u totstake=%llu totslash=%llu\n",
                gpu_r.validator_apply_count, gpu_r.stake_apply_count,
                gpu_r.slash_apply_count, gpu_r.active_validator_count,
                gpu_r.tombstoned_count, gpu_r.jailed_count,
                (unsigned long long)gpu_r.total_active_stake,
                (unsigned long long)gpu_r.total_slashed);
    PASS("Full round matches CPU reference");
}

void test_engine_jailed_excluded()
{
    auto engine = PVMGPUEngine::create();
    EXPECT("jail.engine", engine != nullptr);

    std::vector<ValidatorOp> v_ops;
    v_ops.push_back(make_validator_add(1u, 1000u));
    v_ops.push_back(make_validator_add(2u, 2000u));

    ValidatorOp jail{};
    jail.validator_id = 1u;
    jail.kind = static_cast<uint32_t>(ValidatorOpKind::Jail);
    jail.jail_until_epoch = 10u;
    v_ops.push_back(jail);

    auto desc = make_desc(1u);
    auto h = engine->begin_round(desc);
    engine->push_validator_ops(h, v_ops);
    auto gpu_r = engine->run_until_done(h);
    engine->end_round(h);

    auto state = ref::PVMReferenceState::empty();
    auto cpu_r = ref::run_reference(state, desc, v_ops, {}, {});
    EXPECT("jail.match.active", gpu_r.active_validator_count == cpu_r.active_validator_count);
    EXPECT("jail.match.jailed", gpu_r.jailed_count == cpu_r.jailed_count);
    EXPECT("jail.match.eroot",  std::memcmp(gpu_r.epoch_root, cpu_r.epoch_root, 32) == 0);
    PASS("Jailed validator excluded");
}

void test_engine_redelegate_atomic()
{
    auto engine = PVMGPUEngine::create();
    EXPECT("rd.engine", engine != nullptr);

    std::vector<ValidatorOp> v_ops;
    v_ops.push_back(make_validator_add(1u, 0u));
    v_ops.push_back(make_validator_add(2u, 0u));

    std::vector<StakeOp> s_ops;
    s_ops.push_back(make_bond(100u, 1u, 500u));
    StakeOp rd{};
    rd.delegator_id = 100u;
    rd.validator_id = 2u;
    rd.source_validator_id = 1u;
    rd.amount = 300u;
    rd.kind = static_cast<uint32_t>(StakeOpKind::Redelegate);
    s_ops.push_back(rd);

    auto desc = make_desc(1u);
    auto h = engine->begin_round(desc);
    engine->push_validator_ops(h, v_ops);
    engine->push_stake_ops(h, s_ops);
    auto gpu_r = engine->run_until_done(h);
    engine->end_round(h);

    auto state = ref::PVMReferenceState::empty();
    auto cpu_r = ref::run_reference(state, desc, v_ops, s_ops, {});
    EXPECT("rd.match.total",    gpu_r.total_active_stake == cpu_r.total_active_stake);
    EXPECT("rd.match.eroot",    std::memcmp(gpu_r.epoch_root, cpu_r.epoch_root, 32) == 0);
    PASS("Redelegate atomic");
}

void test_engine_empty_round_deterministic()
{
    auto engine = PVMGPUEngine::create();
    EXPECT("empty.engine", engine != nullptr);

    auto desc = make_desc(1u);
    auto h1 = engine->begin_round(desc);
    auto r1 = engine->run_until_done(h1);
    engine->end_round(h1);
    auto h2 = engine->begin_round(desc);
    auto r2 = engine->run_until_done(h2);
    engine->end_round(h2);

    EXPECT("empty.same.eroot", std::memcmp(r1.epoch_root, r2.epoch_root, 32) == 0);

    bool any_nonzero = false;
    for (auto b : r1.epoch_root) if (b != 0) { any_nonzero = true; break; }
    EXPECT("empty.eroot.nonzero", any_nonzero);

    auto state = ref::PVMReferenceState::empty();
    auto cpu_r = ref::run_reference(state, desc, {}, {}, {});
    EXPECT("empty.match.cpu", std::memcmp(r1.epoch_root, cpu_r.epoch_root, 32) == 0);
    PASS("Empty round deterministic and matches CPU");
}

void test_engine_reward_overflow_safe()
{
    auto engine = PVMGPUEngine::create();
    EXPECT("rew.engine", engine != nullptr);

    std::vector<ValidatorOp> v_ops;
    v_ops.push_back(make_validator_add(1u, 100u));

    std::vector<StakeOp> s_ops;
    s_ops.push_back(make_bond(100u, 1u, 100u));
    // Big reward — exercise saturating math.
    StakeOp rew{};
    rew.validator_id = 1u;
    rew.amount = uint64_t{1} << 60;
    rew.kind = static_cast<uint32_t>(StakeOpKind::Reward);
    s_ops.push_back(rew);

    auto desc = make_desc(1u);
    auto h = engine->begin_round(desc);
    engine->push_validator_ops(h, v_ops);
    engine->push_stake_ops(h, s_ops);
    auto gpu_r = engine->run_until_done(h);
    engine->end_round(h);

    auto state = ref::PVMReferenceState::empty();
    auto cpu_r = ref::run_reference(state, desc, v_ops, s_ops, {});

    // Both must agree even under saturation — that's the contract.
    EXPECT("rew.match.eroot",   std::memcmp(gpu_r.epoch_root, cpu_r.epoch_root, 32) == 0);
    EXPECT("rew.match.rewards", gpu_r.total_rewards == cpu_r.total_rewards);
    PASS("Reward saturation matches CPU");
}

}  // namespace

int main(int /*argc*/, char** /*argv*/)
{
    setvbuf(stdout, nullptr, _IOLBF, 0);
    @autoreleasepool {
        std::printf("[pvm_gpu_engine_test] starting\n");

        test_engine_creates();
        test_engine_full_round_matches_cpu();
        test_engine_jailed_excluded();
        test_engine_redelegate_atomic();
        test_engine_empty_round_deterministic();
        test_engine_reward_overflow_safe();

        std::printf("[pvm_gpu_engine_test] passed=%d failed=%d\n",
                    g_passed, g_failed);
        return g_failed == 0 ? 0 : 1;
    }
}
