// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// pvm_gpu_engine.mm — Metal-backed driver for PVMGPUEngine.
//
// One round = four sequential kernel dispatches in canonical order:
//   1. pvm_validator_set_apply
//   2. pvm_stake_transition
//   3. pvm_slashing_transition
//   4. pvm_epoch_transition
//
// Each dispatch is a single thread (1x1x1) — the kernels do canonical
// in-order traversal of their op streams. This is the minimum
// implementation that satisfies "GPU-resident state + GPU-executed
// canonical transition logic" — i.e. LP-137 GPU-native. Sharding lives
// in a future revision once the determinism contract is locked in.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "lux/pvm/pvm_gpu_engine.hpp"

#include <atomic>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

namespace pvm::gpu {

namespace {

// Try to load a pre-compiled .metallib first. The build pipeline drops
// pvm.metallib next to the .metal sources; loading bytecode is far more
// reliable than the online MSL compiler service (which sometimes returns
// XPC_ERROR_CONNECTION_INTERRUPTED for large kernels).
id<MTLLibrary> load_pvm_metallib(id<MTLDevice> device)
{
    NSError* error = nil;
    std::filesystem::path here = std::filesystem::path(__FILE__).parent_path();
    NSString* exe_path = [[NSBundle mainBundle] executablePath];
    std::filesystem::path exe_dir = exe_path
        ? std::filesystem::path([exe_path UTF8String]).parent_path()
        : std::filesystem::path();
    std::filesystem::path candidates[] = {
        // Build-tree layout (test runs from cmake build dir).
        exe_dir / "src" / "pvm.metallib",
        exe_dir / "pvm.metallib",
        exe_dir.parent_path() / "src" / "pvm.metallib",
        // Source-tree layout (rare).
        here / "pvm.metallib",
        // CWD-relative fallbacks.
        std::filesystem::current_path() / "pvm.metallib",
        std::filesystem::current_path() / "src" / "pvm.metallib",
        std::filesystem::current_path() / "build" / "src" / "pvm.metallib",
        std::filesystem::current_path() / "pvm" / "src" / "pvm.metallib",
        std::filesystem::current_path().parent_path() / "src" / "pvm.metallib",
    };
    for (const auto& p : candidates) {
        if (p.empty() || !std::filesystem::exists(p)) continue;
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:p.c_str()]];
        id<MTLLibrary> lib = [device newLibraryWithURL:url error:&error];
        if (lib) return lib;
        if (error)
            std::fprintf(stderr, "metallib load error %s: %s\n",
                         p.c_str(), [[error localizedDescription] UTF8String]);
    }
    return nil;
}

// Build one MTLLibrary that contains all four PVM kernels. The runtime
// metal compiler tolerates a single large translation unit far better
// than four sequential newLibraryWithSource calls (XPC churn).
id<MTLLibrary> compile_pvm_library(id<MTLDevice> device)
{
    NSError* error = nil;
    std::filesystem::path here = std::filesystem::path(__FILE__).parent_path();
    std::filesystem::path candidates_dir[] = {
        here,
        std::filesystem::current_path(),
        std::filesystem::current_path() / "src",
        std::filesystem::current_path() / "pvm" / "src",
        std::filesystem::current_path().parent_path() / "src",
    };
    auto load_file = [&](const std::filesystem::path& p) -> NSString* {
        if (!std::filesystem::exists(p)) return nil;
        NSString* path = [NSString stringWithUTF8String:p.c_str()];
        return [NSString stringWithContentsOfFile:path
                                         encoding:NSUTF8StringEncoding
                                            error:&error];
    };
    NSString* common = nil;
    NSString* k_v = nil;
    NSString* k_s = nil;
    NSString* k_sl = nil;
    NSString* k_e = nil;
    for (const auto& dir : candidates_dir) {
        common = load_file(dir / "pvm_kernels_common.h.metal");
        k_v    = load_file(dir / "pvm_validator_set.metal");
        k_s    = load_file(dir / "pvm_staking.metal");
        k_sl   = load_file(dir / "pvm_slashing.metal");
        k_e    = load_file(dir / "pvm_transition.metal");
        if (common && k_v && k_s && k_sl && k_e) break;
    }
    if (!common || !k_v || !k_s || !k_sl || !k_e) {
        std::fprintf(stderr, "PVM Metal sources not found near %s\n",
                     here.c_str());
        return nil;
    }
    auto strip_include = [](NSString* src) -> NSString* {
        NSMutableString* out = [NSMutableString string];
        NSArray<NSString*>* lines = [src componentsSeparatedByString:@"\n"];
        for (NSString* line in lines) {
            if ([line containsString:@"pvm_kernels_common.h.metal"]) continue;
            [out appendString:line];
            [out appendString:@"\n"];
        }
        return out;
    };
    NSMutableString* combined = [NSMutableString string];
    [combined appendString:common];
    [combined appendString:@"\n"];
    [combined appendString:strip_include(k_v)];
    [combined appendString:@"\n"];
    [combined appendString:strip_include(k_s)];
    [combined appendString:@"\n"];
    [combined appendString:strip_include(k_sl)];
    [combined appendString:@"\n"];
    [combined appendString:strip_include(k_e)];

    // The AOT metallib (compiled at -O0 via CMake) is the primary path.
    // This runtime fallback only runs if the metallib is missing.
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    opts.languageVersion = MTLLanguageVersion3_0;
    id<MTLLibrary> lib = [device newLibraryWithSource:combined
                                              options:opts
                                                error:&error];
    if (!lib && error)
        std::fprintf(stderr, "PVM Metal compile error: %s\n",
                     [[error localizedDescription] UTF8String]);
    return lib;
}

// Buffer sizes computed from layout constants.
constexpr uint32_t kValidatorSlots = kDefaultValidatorSlots;
constexpr uint32_t kStakeSlots     = kDefaultStakeSlots;
constexpr uint32_t kSlashSlots     = kDefaultSlashSlots;
constexpr uint32_t kMaxOpsPerRound = 4096u;

struct Round {
    PVMRoundHandle handle{};
    PVMRoundDescriptor desc{};

    id<MTLBuffer> desc_buf            = nil;
    id<MTLBuffer> validator_ops_buf   = nil;
    id<MTLBuffer> stake_ops_buf       = nil;
    id<MTLBuffer> slash_ev_buf        = nil;
    id<MTLBuffer> validators_buf      = nil;
    id<MTLBuffer> stake_buf           = nil;
    id<MTLBuffer> slashing_buf        = nil;
    id<MTLBuffer> epoch_buf           = nil;
    id<MTLBuffer> result_buf          = nil;
    id<MTLBuffer> validator_applied_buf = nil;
    id<MTLBuffer> stake_applied_buf     = nil;
    id<MTLBuffer> slash_applied_buf     = nil;
    id<MTLBuffer> total_lo_buf          = nil;
    id<MTLBuffer> total_hi_buf          = nil;
};

class PVMGPUEngineMetal final : public PVMGPUEngine {
public:
    PVMGPUEngineMetal(id<MTLDevice> device,
                      id<MTLCommandQueue> queue,
                      id<MTLComputePipelineState> validator_pso,
                      id<MTLComputePipelineState> stake_pso,
                      id<MTLComputePipelineState> slash_pso,
                      id<MTLComputePipelineState> epoch_pso,
                      NSString* device_name)
        : device_(device)
        , queue_(queue)
        , validator_pso_(validator_pso)
        , stake_pso_(stake_pso)
        , slash_pso_(slash_pso)
        , epoch_pso_(epoch_pso)
        , device_name_str_([device_name UTF8String]) {}

    ~PVMGPUEngineMetal() override {
        if (round_active()) end_round(round_.handle);
    }

    const char* device_name() const override { return device_name_str_.c_str(); }
    bool round_active() const override { return round_.handle.valid(); }

    PVMRoundHandle begin_round(const PVMRoundDescriptor& desc) override {
        std::lock_guard<std::mutex> g(mu_);
        if (round_.handle.valid()) return PVMRoundHandle{0};

        round_ = Round{};
        round_.desc = desc;

        round_.desc_buf            = [device_ newBufferWithLength:sizeof(PVMRoundDescriptor) options:MTLResourceStorageModeShared];
        round_.validator_ops_buf   = [device_ newBufferWithLength:sizeof(ValidatorOp) * kMaxOpsPerRound options:MTLResourceStorageModeShared];
        round_.stake_ops_buf       = [device_ newBufferWithLength:sizeof(StakeOp) * kMaxOpsPerRound options:MTLResourceStorageModeShared];
        round_.slash_ev_buf        = [device_ newBufferWithLength:sizeof(SlashEvidence) * kMaxOpsPerRound options:MTLResourceStorageModeShared];
        round_.validators_buf      = [device_ newBufferWithLength:sizeof(ValidatorSlot) * kValidatorSlots options:MTLResourceStorageModeShared];
        round_.stake_buf           = [device_ newBufferWithLength:sizeof(StakeRecord) * kStakeSlots options:MTLResourceStorageModeShared];
        round_.slashing_buf        = [device_ newBufferWithLength:sizeof(SlashEvidence) * kSlashSlots options:MTLResourceStorageModeShared];
        round_.epoch_buf           = [device_ newBufferWithLength:sizeof(EpochState) options:MTLResourceStorageModeShared];
        round_.result_buf          = [device_ newBufferWithLength:sizeof(PVMTransitionResult) options:MTLResourceStorageModeShared];
        round_.validator_applied_buf = [device_ newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        round_.stake_applied_buf     = [device_ newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        round_.slash_applied_buf     = [device_ newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        round_.total_lo_buf          = [device_ newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        round_.total_hi_buf          = [device_ newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];

        if (!round_.desc_buf || !round_.validator_ops_buf || !round_.stake_ops_buf
            || !round_.slash_ev_buf || !round_.validators_buf || !round_.stake_buf
            || !round_.slashing_buf || !round_.epoch_buf || !round_.result_buf
            || !round_.validator_applied_buf || !round_.stake_applied_buf
            || !round_.slash_applied_buf || !round_.total_lo_buf
            || !round_.total_hi_buf)
            return PVMRoundHandle{0};

        std::memset([round_.validators_buf contents], 0, sizeof(ValidatorSlot) * kValidatorSlots);
        std::memset([round_.stake_buf contents], 0, sizeof(StakeRecord) * kStakeSlots);
        std::memset([round_.slashing_buf contents], 0, sizeof(SlashEvidence) * kSlashSlots);
        std::memset([round_.epoch_buf contents], 0, sizeof(EpochState));
        std::memset([round_.result_buf contents], 0, sizeof(PVMTransitionResult));
        *static_cast<uint32_t*>([round_.validator_applied_buf contents]) = 0;
        *static_cast<uint32_t*>([round_.stake_applied_buf contents]) = 0;
        *static_cast<uint32_t*>([round_.slash_applied_buf contents]) = 0;
        *static_cast<uint32_t*>([round_.total_lo_buf contents]) = 0;
        *static_cast<uint32_t*>([round_.total_hi_buf contents]) = 0;

        // Zero op_count fields — host populates via push_*_ops.
        round_.desc.validator_op_count = 0;
        round_.desc.stake_op_count = 0;
        round_.desc.slash_evidence_count = 0;

        round_.handle = PVMRoundHandle{++next_handle_};
        return round_.handle;
    }

    void push_validator_ops(PVMRoundHandle h, std::span<const ValidatorOp> ops) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h)) return;
        if (ops.empty()) return;
        auto* dst = static_cast<ValidatorOp*>([round_.validator_ops_buf contents]);
        uint32_t cap_left = kMaxOpsPerRound - round_.desc.validator_op_count;
        uint32_t take = std::min<uint32_t>(uint32_t(ops.size()), cap_left);
        std::memcpy(dst + round_.desc.validator_op_count,
                    ops.data(), take * sizeof(ValidatorOp));
        round_.desc.validator_op_count += take;
    }

    void push_stake_ops(PVMRoundHandle h, std::span<const StakeOp> ops) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h)) return;
        if (ops.empty()) return;
        auto* dst = static_cast<StakeOp*>([round_.stake_ops_buf contents]);
        uint32_t cap_left = kMaxOpsPerRound - round_.desc.stake_op_count;
        uint32_t take = std::min<uint32_t>(uint32_t(ops.size()), cap_left);
        std::memcpy(dst + round_.desc.stake_op_count,
                    ops.data(), take * sizeof(StakeOp));
        round_.desc.stake_op_count += take;
    }

    void push_slash_evidence(PVMRoundHandle h, std::span<const SlashEvidence> ev) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h)) return;
        if (ev.empty()) return;
        auto* dst = static_cast<SlashEvidence*>([round_.slash_ev_buf contents]);
        uint32_t cap_left = kMaxOpsPerRound - round_.desc.slash_evidence_count;
        uint32_t take = std::min<uint32_t>(uint32_t(ev.size()), cap_left);
        std::memcpy(dst + round_.desc.slash_evidence_count,
                    ev.data(), take * sizeof(SlashEvidence));
        round_.desc.slash_evidence_count += take;
    }

    PVMTransitionResult run_epoch(PVMRoundHandle h) override {
        return run_until_done(h, 1);
    }

    PVMTransitionResult run_until_done(PVMRoundHandle h, std::size_t /*max_epochs*/) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h)) return PVMTransitionResult{};

        // Push the current desc to the device.
        std::memcpy([round_.desc_buf contents], &round_.desc, sizeof(PVMRoundDescriptor));

        id<MTLCommandBuffer> cmd = [queue_ commandBuffer];

        auto dispatch = [&](id<MTLComputePipelineState> pso,
                            void(^bind)(id<MTLComputeCommandEncoder>)) {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso];
            bind(enc);
            [enc dispatchThreads:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            [enc endEncoding];
        };

        uint32_t validator_count_v = kValidatorSlots;
        uint32_t stake_count_v     = kStakeSlots;
        uint32_t slashing_count_v  = kSlashSlots;

        auto mode = static_cast<PVMTransitionMode>(round_.desc.mode);

        if (mode == PVMTransitionMode::ValidatorSetApply ||
            mode == PVMTransitionMode::FullRound) {
            dispatch(validator_pso_, ^(id<MTLComputeCommandEncoder> enc) {
                [enc setBuffer:round_.desc_buf            offset:0 atIndex:0];
                [enc setBuffer:round_.validator_ops_buf   offset:0 atIndex:1];
                [enc setBuffer:round_.validators_buf      offset:0 atIndex:2];
                [enc setBuffer:round_.validator_applied_buf offset:0 atIndex:3];
                [enc setBytes:&validator_count_v length:sizeof(validator_count_v) atIndex:4];
            });
        }
        if (mode == PVMTransitionMode::StakeTransition ||
            mode == PVMTransitionMode::FullRound) {
            dispatch(stake_pso_, ^(id<MTLComputeCommandEncoder> enc) {
                [enc setBuffer:round_.desc_buf          offset:0 atIndex:0];
                [enc setBuffer:round_.stake_ops_buf     offset:0 atIndex:1];
                [enc setBuffer:round_.validators_buf    offset:0 atIndex:2];
                [enc setBuffer:round_.stake_buf         offset:0 atIndex:3];
                [enc setBuffer:round_.stake_applied_buf offset:0 atIndex:4];
                [enc setBytes:&validator_count_v length:sizeof(validator_count_v) atIndex:5];
                [enc setBytes:&stake_count_v     length:sizeof(stake_count_v)     atIndex:6];
            });
        }
        if (mode == PVMTransitionMode::SlashingTransition ||
            mode == PVMTransitionMode::FullRound) {
            dispatch(slash_pso_, ^(id<MTLComputeCommandEncoder> enc) {
                [enc setBuffer:round_.desc_buf          offset:0 atIndex:0];
                [enc setBuffer:round_.slash_ev_buf      offset:0 atIndex:1];
                [enc setBuffer:round_.validators_buf    offset:0 atIndex:2];
                [enc setBuffer:round_.slashing_buf      offset:0 atIndex:3];
                [enc setBuffer:round_.slash_applied_buf offset:0 atIndex:4];
                [enc setBuffer:round_.total_lo_buf      offset:0 atIndex:5];
                [enc setBuffer:round_.total_hi_buf      offset:0 atIndex:6];
                [enc setBytes:&validator_count_v length:sizeof(validator_count_v) atIndex:7];
                [enc setBytes:&slashing_count_v  length:sizeof(slashing_count_v)  atIndex:8];
            });
        }
        // EpochTransition runs unconditionally — same contract as the CPU reference.
        dispatch(epoch_pso_, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:round_.desc_buf       offset:0 atIndex:0];
            [enc setBuffer:round_.validators_buf offset:0 atIndex:1];
            [enc setBuffer:round_.stake_buf      offset:0 atIndex:2];
            [enc setBuffer:round_.slashing_buf   offset:0 atIndex:3];
            [enc setBuffer:round_.epoch_buf      offset:0 atIndex:4];
            [enc setBuffer:round_.result_buf     offset:0 atIndex:5];
            [enc setBytes:&validator_count_v length:sizeof(validator_count_v) atIndex:6];
            [enc setBytes:&stake_count_v     length:sizeof(stake_count_v)     atIndex:7];
            [enc setBytes:&slashing_count_v  length:sizeof(slashing_count_v)  atIndex:8];
        });

        [cmd commit];
        [cmd waitUntilCompleted];

        // Pull result + counters back. The kernels write applied-counts via
        // separate atomics so we can stamp them onto the result here.
        auto* result = static_cast<PVMTransitionResult*>([round_.result_buf contents]);
        uint32_t v_app = *static_cast<uint32_t*>([round_.validator_applied_buf contents]);
        uint32_t s_app = *static_cast<uint32_t*>([round_.stake_applied_buf contents]);
        uint32_t sl_app= *static_cast<uint32_t*>([round_.slash_applied_buf contents]);
        uint32_t lo    = *static_cast<uint32_t*>([round_.total_lo_buf contents]);
        uint32_t hi    = *static_cast<uint32_t*>([round_.total_hi_buf contents]);
        result->validator_apply_count = v_app;
        result->stake_apply_count = s_app;
        result->slash_apply_count = sl_app;
        result->total_slashed = (uint64_t(hi) << 32) | uint64_t(lo);
        return *result;
    }

    PVMTransitionResult poll_round_result(PVMRoundHandle h) const override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle_const(h)) return PVMTransitionResult{};
        return *static_cast<const PVMTransitionResult*>([round_.result_buf contents]);
    }

    void end_round(PVMRoundHandle h) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h)) return;
        round_ = Round{};
    }

private:
    bool check_handle(PVMRoundHandle h) const {
        return h.valid() && h.opaque == round_.handle.opaque;
    }
    bool check_handle_const(PVMRoundHandle h) const { return check_handle(h); }

    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    id<MTLComputePipelineState> validator_pso_;
    id<MTLComputePipelineState> stake_pso_;
    id<MTLComputePipelineState> slash_pso_;
    id<MTLComputePipelineState> epoch_pso_;
    std::string device_name_str_;
    Round round_;
    uint64_t next_handle_ = 0;
    mutable std::mutex mu_;
};

}  // namespace

std::unique_ptr<PVMGPUEngine> PVMGPUEngine::create() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return nullptr;
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) return nullptr;

        // Prefer a pre-compiled .metallib (build-time AOT compile);
        // fall back to runtime source compilation if it isn't found.
        id<MTLLibrary> lib = load_pvm_metallib(device);
        if (!lib) lib = compile_pvm_library(device);
        if (!lib) return nullptr;

        NSError* err = nil;
        auto fn = [&](NSString* name) -> id<MTLComputePipelineState> {
            id<MTLFunction> f = [lib newFunctionWithName:name];
            if (!f) {
                std::fprintf(stderr, "PVM kernel %s not found in library\n",
                             [name UTF8String]);
                return nil;
            }
            id<MTLComputePipelineState> p = [device newComputePipelineStateWithFunction:f error:&err];
            if (!p && err)
                std::fprintf(stderr, "PSO compile error for %s: %s\n",
                             [name UTF8String], [[err localizedDescription] UTF8String]);
            return p;
        };
        id<MTLComputePipelineState> v_pso  = fn(@"pvm_validator_set_apply");
        id<MTLComputePipelineState> s_pso  = fn(@"pvm_stake_transition");
        id<MTLComputePipelineState> sl_pso = fn(@"pvm_slashing_transition");
        id<MTLComputePipelineState> e_pso  = fn(@"pvm_epoch_transition");
        if (!v_pso || !s_pso || !sl_pso || !e_pso) return nullptr;

        return std::unique_ptr<PVMGPUEngine>(
            new PVMGPUEngineMetal(device, queue, v_pso, s_pso, sl_pso, e_pso,
                                  [device name]));
    }
}

}  // namespace pvm::gpu
