// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// pvm_validator_set.metal — ValidatorSetApply kernel.
//
// Walks validator_ops in canonical order and mutates the validator arena.
// Single-threadgroup execution preserves byte-for-byte determinism with
// the CPU reference. Future revisions can shard once the contract is
// proven; for now the right algorithm is the deterministic one.

#include "pvm_kernels_common.h.metal"

kernel void pvm_validator_set_apply(
    device const PVMRoundDescriptor* desc        [[buffer(0)]],
    device const ValidatorOp*        ops         [[buffer(1)]],
    device ValidatorSlot*            validators  [[buffer(2)]],
    device atomic_uint*              applied_out [[buffer(3)]],
    constant uint&                   validator_count [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0u) return;
    uint applied = 0;
    uint count = desc->validator_op_count;
    for (uint i = 0; i < count; ++i) {
        const device ValidatorOp& op = ops[i];
        switch (op.kind) {
            case kVOpAdd: {
                uint idx = validator_locate(validators, validator_count,
                                            op.validator_id, true);
                if (idx == 0xFFFFFFFFu) break;
                device ValidatorSlot& s = validators[idx];
                s.weight = op.weight;
                for (uint k = 0; k < 48u; ++k) s.bls_pubkey[k] = op.bls_pubkey[k];
                for (uint k = 0; k < 32u; ++k) s.ringtail_pubkey[k] = op.ringtail_pubkey[k];
                for (uint k = 0; k < 32u; ++k) s.mldsa_pubkey[k] = op.mldsa_pubkey[k];
                for (uint k = 0; k < 32u; ++k) s.mldsa_groth16_root[k] = op.mldsa_groth16_root[k];
                s.status = kStatusActive | kStatusPendingAdd;
                s.jail_until_epoch = 0;
                ++applied;
                break;
            }
            case kVOpRemove: {
                uint idx = validator_locate(validators, validator_count,
                                            op.validator_id, false);
                if (idx == 0xFFFFFFFFu) break;
                device ValidatorSlot& s = validators[idx];
                if ((s.status & kStatusTombstoned) != 0u) break;
                s.status |= kStatusPendingDrop;
                s.status &= ~kStatusActive;
                ++applied;
                break;
            }
            case kVOpUpdateWeight: {
                uint idx = validator_locate(validators, validator_count,
                                            op.validator_id, false);
                if (idx == 0xFFFFFFFFu) break;
                device ValidatorSlot& s = validators[idx];
                if ((s.status & kStatusTombstoned) != 0u) break;
                s.weight = op.weight;
                ++applied;
                break;
            }
            case kVOpJail: {
                uint idx = validator_locate(validators, validator_count,
                                            op.validator_id, false);
                if (idx == 0xFFFFFFFFu) break;
                device ValidatorSlot& s = validators[idx];
                if ((s.status & kStatusTombstoned) != 0u) break;
                s.status |= kStatusJailed;
                s.status &= ~kStatusActive;
                if (op.jail_until_epoch > s.jail_until_epoch)
                    s.jail_until_epoch = op.jail_until_epoch;
                ++applied;
                break;
            }
            case kVOpUnjail: {
                uint idx = validator_locate(validators, validator_count,
                                            op.validator_id, false);
                if (idx == 0xFFFFFFFFu) break;
                device ValidatorSlot& s = validators[idx];
                if ((s.status & kStatusTombstoned) != 0u) break;
                if (op.epoch < s.jail_until_epoch) break;
                s.status &= ~kStatusJailed;
                s.status |= kStatusActive;
                s.jail_until_epoch = 0;
                ++applied;
                break;
            }
            case kVOpRotateKeys: {
                uint idx = validator_locate(validators, validator_count,
                                            op.validator_id, false);
                if (idx == 0xFFFFFFFFu) break;
                device ValidatorSlot& s = validators[idx];
                if ((s.status & kStatusTombstoned) != 0u) break;
                for (uint k = 0; k < 48u; ++k) s.bls_pubkey[k] = op.bls_pubkey[k];
                for (uint k = 0; k < 32u; ++k) s.ringtail_pubkey[k] = op.ringtail_pubkey[k];
                for (uint k = 0; k < 32u; ++k) s.mldsa_pubkey[k] = op.mldsa_pubkey[k];
                for (uint k = 0; k < 32u; ++k) s.mldsa_groth16_root[k] = op.mldsa_groth16_root[k];
                ++applied;
                break;
            }
            default:
                break;
        }
    }
    atomic_store_explicit(applied_out, applied, memory_order_relaxed);
}
