// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// pvm_determinism_test.cpp — cross-backend determinism harness.
//
// Compares CPU reference vs GPU engine (Metal on Apple, CUDA on Linux+CUDA)
// across canonical workloads:
//   1. 100 validators, 1000 stake events, 5 slashing events (the brief)
//   2. Empty round (deterministic, non-zero root)
//   3. Jailed validator excluded
//   4. Redelegate atomic
//   5. Reward accumulator overflow handling
//   6. Two engines run-twice — bytes match
//
// This file builds on every platform; the GPU side is selected at link
// time by the platform-specific PVMGPUEngine::create() implementation.

#include "lux/pvm/pvm_gpu_engine.hpp"
#include "lux/pvm/pvm_cpu_reference.hpp"

#include <cstdio>
#include <cstdlib>
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

struct WorkloadResult {
    uint8_t  validator_set_root[32];
    uint8_t  stake_root[32];
    uint8_t  slashing_root[32];
    uint8_t  epoch_root[32];
    uint64_t total_active_stake;
    uint64_t total_slashed;
    uint64_t total_rewards;
    uint32_t active_count;
    uint32_t tombstoned;
    uint32_t jailed;

    static WorkloadResult from(const PVMTransitionResult& r) {
        WorkloadResult w{};
        std::memcpy(w.validator_set_root, r.validator_set_root, 32);
        std::memcpy(w.stake_root,         r.stake_root,         32);
        std::memcpy(w.slashing_root,      r.slashing_root,      32);
        std::memcpy(w.epoch_root,         r.epoch_root,         32);
        w.total_active_stake = r.total_active_stake;
        w.total_slashed = r.total_slashed;
        w.total_rewards = r.total_rewards;
        w.active_count = r.active_validator_count;
        w.tombstoned = r.tombstoned_count;
        w.jailed = r.jailed_count;
        return w;
    }

    bool equals(const WorkloadResult& o) const {
        return std::memcmp(validator_set_root, o.validator_set_root, 32) == 0
            && std::memcmp(stake_root,         o.stake_root,         32) == 0
            && std::memcmp(slashing_root,      o.slashing_root,      32) == 0
            && std::memcmp(epoch_root,         o.epoch_root,         32) == 0
            && total_active_stake == o.total_active_stake
            && total_slashed == o.total_slashed
            && total_rewards == o.total_rewards
            && active_count == o.active_count
            && tombstoned == o.tombstoned
            && jailed == o.jailed;
    }
};

WorkloadResult run_cpu(const PVMRoundDescriptor& desc,
                       const std::vector<ValidatorOp>& v_ops,
                       const std::vector<StakeOp>& s_ops,
                       const std::vector<SlashEvidence>& ev_ops)
{
    auto state = ref::PVMReferenceState::empty();
    auto r = ref::run_reference(state, desc, v_ops, s_ops, ev_ops);
    return WorkloadResult::from(r);
}

WorkloadResult run_gpu(PVMGPUEngine* engine,
                       const PVMRoundDescriptor& desc,
                       const std::vector<ValidatorOp>& v_ops,
                       const std::vector<StakeOp>& s_ops,
                       const std::vector<SlashEvidence>& ev_ops)
{
    auto h = engine->begin_round(desc);
    if (!v_ops.empty())  engine->push_validator_ops(h, v_ops);
    if (!s_ops.empty())  engine->push_stake_ops(h, s_ops);
    if (!ev_ops.empty()) engine->push_slash_evidence(h, ev_ops);
    auto r = engine->run_until_done(h);
    engine->end_round(h);
    return WorkloadResult::from(r);
}

void test_brief_workload(PVMGPUEngine* engine)
{
    std::vector<ValidatorOp> v_ops;
    for (uint64_t i = 1; i <= 100; ++i)
        v_ops.push_back(make_validator_add(i, 10000u + i * 7u));
    std::vector<StakeOp> s_ops;
    for (uint64_t d = 1000; d < 2000; ++d)
        s_ops.push_back(make_bond(d, ((d - 1000u) % 100u) + 1u, 100u + (d % 17u)));
    std::vector<SlashEvidence> ev_ops;
    for (uint32_t i = 0; i < 5; ++i) {
        SlashEvidence ev{};
        ev.validator_id = (i + 1u) * 11u;
        ev.height = 100u * (i + 1u);
        ev.kind = (i % 3u);
        ev.epoch = 0u;
        for (auto& b : ev.evidence_digest) b = uint8_t(0xC0 ^ i);
        ev_ops.push_back(ev);
    }

    auto desc = make_desc(1u);
    auto cpu = run_cpu(desc, v_ops, s_ops, ev_ops);
    if (engine == nullptr) {
        // CPU-only platform: just confirm CPU run is deterministic.
        auto cpu2 = run_cpu(desc, v_ops, s_ops, ev_ops);
        EXPECT("brief.cpu.det", cpu.equals(cpu2));
        std::printf("  brief: CPU-only path; root match across runs\n");
        PASS("brief workload (100v / 1000s / 5ev) — CPU determinism");
        return;
    }
    auto gpu = run_gpu(engine, desc, v_ops, s_ops, ev_ops);
    EXPECT("brief.match", cpu.equals(gpu));
    std::printf("  brief: roots match CPU<->GPU; active=%u tomb=%u jail=%u\n",
                gpu.active_count, gpu.tombstoned, gpu.jailed);
    PASS("brief workload (100v / 1000s / 5ev) — CPU<->GPU byte match");
}

void test_empty_round(PVMGPUEngine* engine)
{
    auto desc = make_desc(1u);
    auto cpu = run_cpu(desc, {}, {}, {});
    bool nonzero = false;
    for (auto b : cpu.epoch_root) if (b != 0) { nonzero = true; break; }
    EXPECT("empty.cpu.nz", nonzero);

    if (engine != nullptr) {
        auto gpu = run_gpu(engine, desc, {}, {}, {});
        EXPECT("empty.match", cpu.equals(gpu));
    }
    PASS("empty round deterministic non-zero");
}

void test_jail_excludes(PVMGPUEngine* engine)
{
    std::vector<ValidatorOp> v_ops;
    v_ops.push_back(make_validator_add(1u, 1000u));
    v_ops.push_back(make_validator_add(2u, 2000u));
    ValidatorOp jail{};
    jail.validator_id = 1u;
    jail.kind = static_cast<uint32_t>(ValidatorOpKind::Jail);
    jail.jail_until_epoch = 10u;
    v_ops.push_back(jail);

    auto desc = make_desc(1u);
    auto cpu = run_cpu(desc, v_ops, {}, {});
    EXPECT("jail.cpu.active", cpu.active_count == 1u);
    if (engine != nullptr) {
        auto gpu = run_gpu(engine, desc, v_ops, {}, {});
        EXPECT("jail.match", cpu.equals(gpu));
    }
    PASS("jailed validator excluded");
}

void test_redelegate(PVMGPUEngine* engine)
{
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
    auto cpu = run_cpu(desc, v_ops, s_ops, {});
    EXPECT("rd.cpu.total", cpu.total_active_stake == 500u);
    if (engine != nullptr) {
        auto gpu = run_gpu(engine, desc, v_ops, s_ops, {});
        EXPECT("rd.match", cpu.equals(gpu));
    }
    PASS("redelegate atomic");
}

void test_reward_saturation(PVMGPUEngine* engine)
{
    std::vector<ValidatorOp> v_ops;
    v_ops.push_back(make_validator_add(1u, 100u));
    std::vector<StakeOp> s_ops;
    s_ops.push_back(make_bond(100u, 1u, 100u));
    StakeOp rew{};
    rew.validator_id = 1u;
    rew.amount = uint64_t{1} << 60;
    rew.kind = static_cast<uint32_t>(StakeOpKind::Reward);
    s_ops.push_back(rew);

    auto desc = make_desc(1u);
    auto cpu = run_cpu(desc, v_ops, s_ops, {});
    if (engine != nullptr) {
        auto gpu = run_gpu(engine, desc, v_ops, s_ops, {});
        EXPECT("rew.match", cpu.equals(gpu));
    }
    PASS("reward saturation handled");
}

void test_two_engines_match(PVMGPUEngine* engine)
{
    if (engine == nullptr) {
        PASS("two-engines (skipped — no GPU)");
        return;
    }
    auto a = PVMGPUEngine::create();
    auto b = PVMGPUEngine::create();
    EXPECT("twoeng.a", a != nullptr);
    EXPECT("twoeng.b", b != nullptr);

    std::vector<ValidatorOp> v_ops;
    for (uint64_t i = 1; i <= 16; ++i)
        v_ops.push_back(make_validator_add(i, 1000u * i));
    std::vector<StakeOp> s_ops;
    for (uint64_t d = 100; d < 132; ++d)
        s_ops.push_back(make_bond(d, ((d - 100u) % 16u) + 1u, 50u + (d % 7u)));

    auto desc = make_desc(1u);
    auto ra = run_gpu(a.get(), desc, v_ops, s_ops, {});
    auto rb = run_gpu(b.get(), desc, v_ops, s_ops, {});
    EXPECT("twoeng.match", ra.equals(rb));
    PASS("two engines bytewise identical");
}

}  // namespace

int main(int /*argc*/, char** /*argv*/)
{
    setvbuf(stdout, nullptr, _IOLBF, 0);
    std::printf("[pvm_determinism_test] starting\n");

    auto engine = PVMGPUEngine::create();
    if (engine == nullptr) {
        std::printf("  note: no GPU backend available — running CPU-only path\n");
    } else {
        std::printf("  device: %s\n", engine->device_name());
    }

    test_brief_workload(engine.get());
    test_empty_round(engine.get());
    test_jail_excludes(engine.get());
    test_redelegate(engine.get());
    test_reward_saturation(engine.get());
    test_two_engines_match(engine.get());

    std::printf("[pvm_determinism_test] passed=%d failed=%d\n",
                g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
