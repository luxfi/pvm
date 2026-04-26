// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// pvm_cuda_launchers.cu — host-side <<<1,1>>> launchers for the four PVM
// CUDA kernel entry points. Kept in its own translation unit so the host
// driver (pvm_gpu_engine_cuda.cpp) stays pure C++ and can be compiled by
// the host C++ compiler without nvcc.

#include "pvm_kernels_common.cuh"

#include "lux/pvm/pvm_gpu_layout.hpp"

namespace pvm::gpu {

// CUDA kernel forward decls — defined in the four .cu files.
namespace cuda_decl {
extern "C" __global__ void pvm_validator_set_apply(
    const ::pvm::cuda::PVMRoundDescriptor*, const ::pvm::cuda::ValidatorOp*,
    ::pvm::cuda::ValidatorSlot*, uint32_t*, uint32_t);
extern "C" __global__ void pvm_stake_transition(
    const ::pvm::cuda::PVMRoundDescriptor*, const ::pvm::cuda::StakeOp*,
    ::pvm::cuda::ValidatorSlot*, ::pvm::cuda::StakeRecord*,
    uint32_t*, uint32_t, uint32_t);
extern "C" __global__ void pvm_slashing_transition(
    const ::pvm::cuda::PVMRoundDescriptor*, const ::pvm::cuda::SlashEvidence*,
    ::pvm::cuda::ValidatorSlot*, ::pvm::cuda::SlashEvidence*,
    uint32_t*, uint64_t*, uint32_t, uint32_t);
extern "C" __global__ void pvm_epoch_transition(
    const ::pvm::cuda::PVMRoundDescriptor*, ::pvm::cuda::ValidatorSlot*,
    ::pvm::cuda::StakeRecord*, ::pvm::cuda::SlashEvidence*,
    ::pvm::cuda::EpochState*, ::pvm::cuda::PVMTransitionResult*,
    uint32_t, uint32_t, uint32_t);
}  // namespace cuda_decl

// The pvm::gpu host structs and pvm::cuda device structs share the same
// byte layout (they are duplicates with the same alignas/members). We
// reinterpret_cast at the API boundary so the host driver can use
// pvm::gpu types throughout.

void launch_pvm_validator_set_apply(
    const PVMRoundDescriptor* desc, const ValidatorOp* ops,
    ValidatorSlot* validators, uint32_t* applied_out, uint32_t validator_count)
{
    cuda_decl::pvm_validator_set_apply<<<1, 1>>>(
        reinterpret_cast<const ::pvm::cuda::PVMRoundDescriptor*>(desc),
        reinterpret_cast<const ::pvm::cuda::ValidatorOp*>(ops),
        reinterpret_cast<::pvm::cuda::ValidatorSlot*>(validators),
        applied_out, validator_count);
}

void launch_pvm_stake_transition(
    const PVMRoundDescriptor* desc, const StakeOp* ops,
    ValidatorSlot* validators, StakeRecord* stake,
    uint32_t* applied_out, uint32_t validator_count, uint32_t stake_count)
{
    cuda_decl::pvm_stake_transition<<<1, 1>>>(
        reinterpret_cast<const ::pvm::cuda::PVMRoundDescriptor*>(desc),
        reinterpret_cast<const ::pvm::cuda::StakeOp*>(ops),
        reinterpret_cast<::pvm::cuda::ValidatorSlot*>(validators),
        reinterpret_cast<::pvm::cuda::StakeRecord*>(stake),
        applied_out, validator_count, stake_count);
}

void launch_pvm_slashing_transition(
    const PVMRoundDescriptor* desc, const SlashEvidence* evidence,
    ValidatorSlot* validators, SlashEvidence* slashing,
    uint32_t* applied_out, uint64_t* total_slashed_out,
    uint32_t validator_count, uint32_t slashing_count)
{
    cuda_decl::pvm_slashing_transition<<<1, 1>>>(
        reinterpret_cast<const ::pvm::cuda::PVMRoundDescriptor*>(desc),
        reinterpret_cast<const ::pvm::cuda::SlashEvidence*>(evidence),
        reinterpret_cast<::pvm::cuda::ValidatorSlot*>(validators),
        reinterpret_cast<::pvm::cuda::SlashEvidence*>(slashing),
        applied_out, total_slashed_out, validator_count, slashing_count);
}

void launch_pvm_epoch_transition(
    const PVMRoundDescriptor* desc, ValidatorSlot* validators, StakeRecord* stake,
    SlashEvidence* slashing, EpochState* epoch, PVMTransitionResult* result,
    uint32_t validator_count, uint32_t stake_count, uint32_t slashing_count)
{
    cuda_decl::pvm_epoch_transition<<<1, 1>>>(
        reinterpret_cast<const ::pvm::cuda::PVMRoundDescriptor*>(desc),
        reinterpret_cast<::pvm::cuda::ValidatorSlot*>(validators),
        reinterpret_cast<::pvm::cuda::StakeRecord*>(stake),
        reinterpret_cast<::pvm::cuda::SlashEvidence*>(slashing),
        reinterpret_cast<::pvm::cuda::EpochState*>(epoch),
        reinterpret_cast<::pvm::cuda::PVMTransitionResult*>(result),
        validator_count, stake_count, slashing_count);
}

}  // namespace pvm::gpu
