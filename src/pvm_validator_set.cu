// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// pvm_validator_set.cu — CUDA peer of pvm_validator_set.metal.

#include "pvm_kernels_common.cuh"

namespace pvm::cuda {

extern "C" __global__ void pvm_validator_set_apply(
    const PVMRoundDescriptor* desc,
    const ValidatorOp*        ops,
    ValidatorSlot*            validators,
    uint32_t*                 applied_out,
    uint32_t                  validator_count)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    uint32_t applied = 0;
    uint32_t count = desc->validator_op_count;
    for (uint32_t i = 0; i < count; ++i) {
        const ValidatorOp& op = ops[i];
        switch (op.kind) {
            case kVOpAdd: {
                uint32_t idx = validator_locate(validators, validator_count,
                                                op.validator_id, true);
                if (idx == 0xFFFFFFFFu) break;
                ValidatorSlot& s = validators[idx];
                s.weight = op.weight;
                for (uint32_t k = 0; k < 48u; ++k) s.bls_pubkey[k] = op.bls_pubkey[k];
                for (uint32_t k = 0; k < 32u; ++k) s.ringtail_pubkey[k] = op.ringtail_pubkey[k];
                for (uint32_t k = 0; k < 32u; ++k) s.mldsa_pubkey[k] = op.mldsa_pubkey[k];
                for (uint32_t k = 0; k < 32u; ++k) s.mldsa_groth16_root[k] = op.mldsa_groth16_root[k];
                s.status = kStatusActive | kStatusPendingAdd;
                s.jail_until_epoch = 0;
                ++applied;
                break;
            }
            case kVOpRemove: {
                uint32_t idx = validator_locate(validators, validator_count,
                                                op.validator_id, false);
                if (idx == 0xFFFFFFFFu) break;
                ValidatorSlot& s = validators[idx];
                if ((s.status & kStatusTombstoned) != 0u) break;
                s.status |= kStatusPendingDrop;
                s.status &= ~kStatusActive;
                ++applied;
                break;
            }
            case kVOpUpdateWeight: {
                uint32_t idx = validator_locate(validators, validator_count,
                                                op.validator_id, false);
                if (idx == 0xFFFFFFFFu) break;
                ValidatorSlot& s = validators[idx];
                if ((s.status & kStatusTombstoned) != 0u) break;
                s.weight = op.weight;
                ++applied;
                break;
            }
            case kVOpJail: {
                uint32_t idx = validator_locate(validators, validator_count,
                                                op.validator_id, false);
                if (idx == 0xFFFFFFFFu) break;
                ValidatorSlot& s = validators[idx];
                if ((s.status & kStatusTombstoned) != 0u) break;
                s.status |= kStatusJailed;
                s.status &= ~kStatusActive;
                if (op.jail_until_epoch > s.jail_until_epoch)
                    s.jail_until_epoch = op.jail_until_epoch;
                ++applied;
                break;
            }
            case kVOpUnjail: {
                uint32_t idx = validator_locate(validators, validator_count,
                                                op.validator_id, false);
                if (idx == 0xFFFFFFFFu) break;
                ValidatorSlot& s = validators[idx];
                if ((s.status & kStatusTombstoned) != 0u) break;
                if (op.epoch < s.jail_until_epoch) break;
                s.status &= ~kStatusJailed;
                s.status |= kStatusActive;
                s.jail_until_epoch = 0;
                ++applied;
                break;
            }
            case kVOpRotateKeys: {
                uint32_t idx = validator_locate(validators, validator_count,
                                                op.validator_id, false);
                if (idx == 0xFFFFFFFFu) break;
                ValidatorSlot& s = validators[idx];
                if ((s.status & kStatusTombstoned) != 0u) break;
                for (uint32_t k = 0; k < 48u; ++k) s.bls_pubkey[k] = op.bls_pubkey[k];
                for (uint32_t k = 0; k < 32u; ++k) s.ringtail_pubkey[k] = op.ringtail_pubkey[k];
                for (uint32_t k = 0; k < 32u; ++k) s.mldsa_pubkey[k] = op.mldsa_pubkey[k];
                for (uint32_t k = 0; k < 32u; ++k) s.mldsa_groth16_root[k] = op.mldsa_groth16_root[k];
                ++applied;
                break;
            }
            default: break;
        }
    }
    *applied_out = applied;
}

}  // namespace pvm::cuda
