// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// pvm_gpu_engine_cuda.cpp — CUDA-backed driver for PVMGPUEngine.
//
// Mirrors the Metal driver: one round = four sequential kernel launches.
// Each launch is <<<1, 1>>> — single-thread canonical traversal preserves
// byte-for-byte determinism with the CPU reference and the Metal driver.
// Future revisions can shard once the determinism contract is locked in.

#include "lux/pvm/pvm_gpu_engine.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

namespace pvm::gpu {

// Forward-declare CUDA kernel entry points (defined in pvm_*.cu).
namespace cuda { struct PVMRoundDescriptor; }

extern "C" {

void pvm_validator_set_apply(
    const PVMRoundDescriptor* desc,
    const ValidatorOp*        ops,
    ValidatorSlot*            validators,
    uint32_t*                 applied_out,
    uint32_t                  validator_count);

void pvm_stake_transition(
    const PVMRoundDescriptor* desc,
    const StakeOp*            ops,
    ValidatorSlot*            validators,
    StakeRecord*              stake,
    uint32_t*                 applied_out,
    uint32_t                  validator_count,
    uint32_t                  stake_count);

void pvm_slashing_transition(
    const PVMRoundDescriptor* desc,
    const SlashEvidence*      evidence,
    ValidatorSlot*            validators,
    SlashEvidence*            slashing,
    uint32_t*                 applied_out,
    uint64_t*                 total_slashed_out,
    uint32_t                  validator_count,
    uint32_t                  slashing_count);

void pvm_epoch_transition(
    const PVMRoundDescriptor* desc,
    ValidatorSlot*            validators,
    StakeRecord*              stake,
    SlashEvidence*            slashing,
    EpochState*               epoch,
    PVMTransitionResult*      result,
    uint32_t                  validator_count,
    uint32_t                  stake_count,
    uint32_t                  slashing_count);

}  // extern "C"

namespace {

// Kernel launchers — these must be implemented in a .cu translation unit.
// We declare them here as host-side launchers and define them as a small
// shim that forwards to <<<1,1>>> launches in the .cu sources.
//
// To keep this file pure C++ (no nvcc), the CMake build links the kernels
// in via separate .cu compilation units; the launchers are defined in
// pvm_cuda_launchers.cu (a tiny wrapper file produced by CMake from
// pvm_gpu_engine_cuda.cu).

extern void launch_pvm_validator_set_apply(
    const PVMRoundDescriptor*, const ValidatorOp*,
    ValidatorSlot*, uint32_t*, uint32_t);
extern void launch_pvm_stake_transition(
    const PVMRoundDescriptor*, const StakeOp*,
    ValidatorSlot*, StakeRecord*, uint32_t*, uint32_t, uint32_t);
extern void launch_pvm_slashing_transition(
    const PVMRoundDescriptor*, const SlashEvidence*,
    ValidatorSlot*, SlashEvidence*, uint32_t*, uint64_t*,
    uint32_t, uint32_t);
extern void launch_pvm_epoch_transition(
    const PVMRoundDescriptor*, ValidatorSlot*, StakeRecord*,
    SlashEvidence*, EpochState*, PVMTransitionResult*,
    uint32_t, uint32_t, uint32_t);

constexpr uint32_t kValidatorSlots = kDefaultValidatorSlots;
constexpr uint32_t kStakeSlots     = kDefaultStakeSlots;
constexpr uint32_t kSlashSlots     = kDefaultSlashSlots;
constexpr uint32_t kMaxOpsPerRound = 4096u;

#define CUDA_CHECK(expr) do {                                       \
        cudaError_t e = (expr);                                     \
        if (e != cudaSuccess) {                                     \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",    \
                         #expr, __FILE__, __LINE__,                 \
                         cudaGetErrorString(e));                    \
            return;                                                 \
        }                                                           \
    } while (0)

struct Round {
    PVMRoundHandle handle{};
    PVMRoundDescriptor desc{};

    PVMRoundDescriptor* d_desc = nullptr;
    ValidatorOp*        d_validator_ops = nullptr;
    StakeOp*            d_stake_ops = nullptr;
    SlashEvidence*      d_slash_ev = nullptr;
    ValidatorSlot*      d_validators = nullptr;
    StakeRecord*        d_stake = nullptr;
    SlashEvidence*      d_slashing = nullptr;
    EpochState*         d_epoch = nullptr;
    PVMTransitionResult* d_result = nullptr;
    uint32_t*           d_v_applied = nullptr;
    uint32_t*           d_s_applied = nullptr;
    uint32_t*           d_sl_applied = nullptr;
    uint64_t*           d_total_slashed = nullptr;
};

class PVMGPUEngineCuda final : public PVMGPUEngine {
public:
    PVMGPUEngineCuda() {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
            device_name_str_ = prop.name;
        } else {
            device_name_str_ = "cuda";
        }
    }
    ~PVMGPUEngineCuda() override {
        if (round_active()) end_round(round_.handle);
    }

    const char* device_name() const override { return device_name_str_.c_str(); }
    bool round_active() const override { return round_.handle.valid(); }

    PVMRoundHandle begin_round(const PVMRoundDescriptor& desc) override {
        std::lock_guard<std::mutex> g(mu_);
        if (round_.handle.valid()) return PVMRoundHandle{0};
        round_ = Round{};
        round_.desc = desc;
        round_.desc.validator_op_count = 0;
        round_.desc.stake_op_count = 0;
        round_.desc.slash_evidence_count = 0;

        auto alloc = [](void** p, size_t n) -> bool {
            if (cudaMalloc(p, n) != cudaSuccess) return false;
            return cudaMemset(*p, 0, n) == cudaSuccess;
        };

        if (!alloc((void**)&round_.d_desc, sizeof(PVMRoundDescriptor))
            || !alloc((void**)&round_.d_validator_ops, sizeof(ValidatorOp) * kMaxOpsPerRound)
            || !alloc((void**)&round_.d_stake_ops, sizeof(StakeOp) * kMaxOpsPerRound)
            || !alloc((void**)&round_.d_slash_ev, sizeof(SlashEvidence) * kMaxOpsPerRound)
            || !alloc((void**)&round_.d_validators, sizeof(ValidatorSlot) * kValidatorSlots)
            || !alloc((void**)&round_.d_stake, sizeof(StakeRecord) * kStakeSlots)
            || !alloc((void**)&round_.d_slashing, sizeof(SlashEvidence) * kSlashSlots)
            || !alloc((void**)&round_.d_epoch, sizeof(EpochState))
            || !alloc((void**)&round_.d_result, sizeof(PVMTransitionResult))
            || !alloc((void**)&round_.d_v_applied, sizeof(uint32_t))
            || !alloc((void**)&round_.d_s_applied, sizeof(uint32_t))
            || !alloc((void**)&round_.d_sl_applied, sizeof(uint32_t))
            || !alloc((void**)&round_.d_total_slashed, sizeof(uint64_t))) {
            return PVMRoundHandle{0};
        }

        round_.handle = PVMRoundHandle{++next_handle_};
        return round_.handle;
    }

    void push_validator_ops(PVMRoundHandle h, std::span<const ValidatorOp> ops) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h) || ops.empty()) return;
        uint32_t cap_left = kMaxOpsPerRound - round_.desc.validator_op_count;
        uint32_t take = std::min<uint32_t>(uint32_t(ops.size()), cap_left);
        cudaMemcpy(round_.d_validator_ops + round_.desc.validator_op_count,
                   ops.data(), take * sizeof(ValidatorOp), cudaMemcpyHostToDevice);
        round_.desc.validator_op_count += take;
    }

    void push_stake_ops(PVMRoundHandle h, std::span<const StakeOp> ops) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h) || ops.empty()) return;
        uint32_t cap_left = kMaxOpsPerRound - round_.desc.stake_op_count;
        uint32_t take = std::min<uint32_t>(uint32_t(ops.size()), cap_left);
        cudaMemcpy(round_.d_stake_ops + round_.desc.stake_op_count,
                   ops.data(), take * sizeof(StakeOp), cudaMemcpyHostToDevice);
        round_.desc.stake_op_count += take;
    }

    void push_slash_evidence(PVMRoundHandle h, std::span<const SlashEvidence> ev) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h) || ev.empty()) return;
        uint32_t cap_left = kMaxOpsPerRound - round_.desc.slash_evidence_count;
        uint32_t take = std::min<uint32_t>(uint32_t(ev.size()), cap_left);
        cudaMemcpy(round_.d_slash_ev + round_.desc.slash_evidence_count,
                   ev.data(), take * sizeof(SlashEvidence), cudaMemcpyHostToDevice);
        round_.desc.slash_evidence_count += take;
    }

    PVMTransitionResult run_epoch(PVMRoundHandle h) override {
        return run_until_done(h, 1);
    }

    PVMTransitionResult run_until_done(PVMRoundHandle h, std::size_t /*max_epochs*/) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h)) return PVMTransitionResult{};

        cudaMemcpy(round_.d_desc, &round_.desc, sizeof(PVMRoundDescriptor),
                   cudaMemcpyHostToDevice);

        auto mode = static_cast<PVMTransitionMode>(round_.desc.mode);
        if (mode == PVMTransitionMode::ValidatorSetApply ||
            mode == PVMTransitionMode::FullRound) {
            launch_pvm_validator_set_apply(
                round_.d_desc, round_.d_validator_ops,
                round_.d_validators, round_.d_v_applied, kValidatorSlots);
        }
        if (mode == PVMTransitionMode::StakeTransition ||
            mode == PVMTransitionMode::FullRound) {
            launch_pvm_stake_transition(
                round_.d_desc, round_.d_stake_ops,
                round_.d_validators, round_.d_stake,
                round_.d_s_applied, kValidatorSlots, kStakeSlots);
        }
        if (mode == PVMTransitionMode::SlashingTransition ||
            mode == PVMTransitionMode::FullRound) {
            launch_pvm_slashing_transition(
                round_.d_desc, round_.d_slash_ev,
                round_.d_validators, round_.d_slashing,
                round_.d_sl_applied, round_.d_total_slashed,
                kValidatorSlots, kSlashSlots);
        }
        launch_pvm_epoch_transition(
            round_.d_desc, round_.d_validators, round_.d_stake,
            round_.d_slashing, round_.d_epoch, round_.d_result,
            kValidatorSlots, kStakeSlots, kSlashSlots);

        cudaDeviceSynchronize();

        PVMTransitionResult result{};
        cudaMemcpy(&result, round_.d_result, sizeof(PVMTransitionResult),
                   cudaMemcpyDeviceToHost);
        uint32_t v_app = 0, s_app = 0, sl_app = 0;
        uint64_t total_slashed = 0;
        cudaMemcpy(&v_app,  round_.d_v_applied,  sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&s_app,  round_.d_s_applied,  sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&sl_app, round_.d_sl_applied, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&total_slashed, round_.d_total_slashed, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        result.validator_apply_count = v_app;
        result.stake_apply_count     = s_app;
        result.slash_apply_count     = sl_app;
        result.total_slashed         = total_slashed;
        return result;
    }

    PVMTransitionResult poll_round_result(PVMRoundHandle h) const override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle_const(h)) return PVMTransitionResult{};
        PVMTransitionResult result{};
        cudaMemcpy(&result, round_.d_result, sizeof(PVMTransitionResult),
                   cudaMemcpyDeviceToHost);
        return result;
    }

    void end_round(PVMRoundHandle h) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h)) return;
        cudaFree(round_.d_desc);
        cudaFree(round_.d_validator_ops);
        cudaFree(round_.d_stake_ops);
        cudaFree(round_.d_slash_ev);
        cudaFree(round_.d_validators);
        cudaFree(round_.d_stake);
        cudaFree(round_.d_slashing);
        cudaFree(round_.d_epoch);
        cudaFree(round_.d_result);
        cudaFree(round_.d_v_applied);
        cudaFree(round_.d_s_applied);
        cudaFree(round_.d_sl_applied);
        cudaFree(round_.d_total_slashed);
        round_ = Round{};
    }

private:
    bool check_handle(PVMRoundHandle h) const {
        return h.valid() && h.opaque == round_.handle.opaque;
    }
    bool check_handle_const(PVMRoundHandle h) const { return check_handle(h); }

    std::string device_name_str_;
    Round round_;
    uint64_t next_handle_ = 0;
    mutable std::mutex mu_;
};

}  // namespace

#if !defined(__APPLE__)
std::unique_ptr<PVMGPUEngine> PVMGPUEngine::create() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n <= 0) return nullptr;
    return std::unique_ptr<PVMGPUEngine>(new PVMGPUEngineCuda());
}
#endif

}  // namespace pvm::gpu
