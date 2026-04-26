// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// pvm_staking.metal — StakeTransition kernel.
//
// Single-threadgroup, canonical-order traversal of stake_ops. Mirrors the
// CPU reference exactly. Reward distribution walks the entire stake arena
// for the targeted validator.

#include "pvm_kernels_common.h.metal"

kernel void pvm_stake_transition(
    device const PVMRoundDescriptor* desc        [[buffer(0)]],
    device const StakeOp*            ops         [[buffer(1)]],
    device ValidatorSlot*            validators  [[buffer(2)]],
    device StakeRecord*              stake       [[buffer(3)]],
    device atomic_uint*              applied_out [[buffer(4)]],
    constant uint&                   validator_count [[buffer(5)]],
    constant uint&                   stake_count     [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0u) return;
    uint applied = 0;
    uint count = desc->stake_op_count;
    for (uint i = 0; i < count; ++i) {
        const device StakeOp& op = ops[i];
        switch (op.kind) {
            case kSOpBond: {
                uint v_idx = validator_locate(validators, validator_count,
                                              op.validator_id, false);
                if (v_idx == 0xFFFFFFFFu) break;
                device ValidatorSlot& v = validators[v_idx];
                if ((v.status & kStatusTombstoned) != 0u) break;

                uint s_idx = stake_record_locate(stake, stake_count,
                                                 op.delegator_id,
                                                 op.validator_id, true);
                if (s_idx == 0xFFFFFFFFu) break;
                device StakeRecord& s = stake[s_idx];
                s.amount = sat_add_u64(s.amount, op.amount);
                if (op.lock_until_epoch > s.lock_until_epoch)
                    s.lock_until_epoch = op.lock_until_epoch;
                if (s.epoch_bonded == 0) s.epoch_bonded = op.epoch;
                s.status = kStakeStatusActive;
                v.weight = sat_add_u64(v.weight, op.amount);
                ++applied;
                break;
            }
            case kSOpUnbond: {
                uint s_idx = stake_record_locate(stake, stake_count,
                                                 op.delegator_id,
                                                 op.validator_id, false);
                if (s_idx == 0xFFFFFFFFu) break;
                device StakeRecord& s = stake[s_idx];
                if (s.status != kStakeStatusActive) break;
                if (op.epoch < s.lock_until_epoch) break;
                ulong amt = (op.amount < s.amount) ? op.amount : s.amount;
                s.amount = sat_sub_u64(s.amount, amt);
                s.epoch_unbonded = op.epoch;
                s.status = (s.amount == 0) ? kStakeStatusRetired : kStakeStatusUnbonding;

                uint v_idx = validator_locate(validators, validator_count,
                                              op.validator_id, false);
                if (v_idx != 0xFFFFFFFFu) {
                    device ValidatorSlot& v = validators[v_idx];
                    v.weight = sat_sub_u64(v.weight, amt);
                }
                ++applied;
                break;
            }
            case kSOpDelegate: {
                uint v_idx = validator_locate(validators, validator_count,
                                              op.validator_id, false);
                if (v_idx == 0xFFFFFFFFu) break;
                device ValidatorSlot& v = validators[v_idx];
                if ((v.status & kStatusTombstoned) != 0u) break;
                if ((v.status & kStatusJailed) != 0u) break;

                uint s_idx = stake_record_locate(stake, stake_count,
                                                 op.delegator_id,
                                                 op.validator_id, true);
                if (s_idx == 0xFFFFFFFFu) break;
                device StakeRecord& s = stake[s_idx];
                s.amount = sat_add_u64(s.amount, op.amount);
                s.status = kStakeStatusActive;
                if (s.epoch_bonded == 0) s.epoch_bonded = op.epoch;
                v.weight = sat_add_u64(v.weight, op.amount);
                ++applied;
                break;
            }
            case kSOpRedelegate: {
                if (op.source_validator_id == op.validator_id) break;

                uint src_idx = stake_record_locate(stake, stake_count,
                                                   op.delegator_id,
                                                   op.source_validator_id, false);
                if (src_idx == 0xFFFFFFFFu) break;
                device StakeRecord& src = stake[src_idx];
                if (src.status != kStakeStatusActive) break;
                if (op.epoch < src.lock_until_epoch) break;

                uint v_dst_idx = validator_locate(validators, validator_count,
                                                  op.validator_id, false);
                if (v_dst_idx == 0xFFFFFFFFu) break;
                device ValidatorSlot& v_dst = validators[v_dst_idx];
                if ((v_dst.status & kStatusTombstoned) != 0u) break;

                ulong amt = (op.amount < src.amount) ? op.amount : src.amount;
                src.amount = sat_sub_u64(src.amount, amt);
                if (src.amount == 0) src.status = kStakeStatusRetired;

                uint v_src_idx = validator_locate(validators, validator_count,
                                                  op.source_validator_id, false);
                if (v_src_idx != 0xFFFFFFFFu) {
                    device ValidatorSlot& v_src = validators[v_src_idx];
                    v_src.weight = sat_sub_u64(v_src.weight, amt);
                }

                uint dst_idx = stake_record_locate(stake, stake_count,
                                                   op.delegator_id,
                                                   op.validator_id, true);
                if (dst_idx == 0xFFFFFFFFu) break;
                device StakeRecord& dst = stake[dst_idx];
                dst.amount = sat_add_u64(dst.amount, amt);
                dst.status = kStakeStatusActive;
                if (dst.epoch_bonded == 0) dst.epoch_bonded = op.epoch;
                v_dst.weight = sat_add_u64(v_dst.weight, amt);
                ++applied;
                break;
            }
            case kSOpReward: {
                uint v_idx = validator_locate(validators, validator_count,
                                              op.validator_id, false);
                if (v_idx == 0xFFFFFFFFu) break;
                device ValidatorSlot& v = validators[v_idx];
                if (v.weight == 0) break;
                ulong scaled = (op.amount > 0xFFFFFFFFFFFFFFFFUL / kRewardScale)
                    ? 0xFFFFFFFFFFFFFFFFUL
                    : op.amount * kRewardScale;
                ulong per_unit = scaled / v.weight;
                if (per_unit == 0) break;
                for (uint si = 0; si < stake_count; ++si) {
                    device StakeRecord& s = stake[si];
                    if (s.status != kStakeStatusActive) continue;
                    if (s.validator_id != op.validator_id) continue;
                    ulong delta = (s.amount > 0xFFFFFFFFFFFFFFFFUL / per_unit)
                        ? 0xFFFFFFFFFFFFFFFFUL
                        : s.amount * per_unit;
                    s.reward_accumulator = sat_add_u64(s.reward_accumulator, delta);
                }
                ++applied;
                break;
            }
            case kSOpCommission: {
                uint v_idx = validator_locate(validators, validator_count,
                                              op.validator_id, false);
                if (v_idx == 0xFFFFFFFFu) break;
                if (op.commission_bps > 10000u) break;
                uint s_idx = stake_record_locate(stake, stake_count,
                                                 op.validator_id,
                                                 op.validator_id, false);
                if (s_idx == 0xFFFFFFFFu) break;
                stake[s_idx].commission_bps = op.commission_bps;
                ++applied;
                break;
            }
            default:
                break;
        }
    }
    atomic_store_explicit(applied_out, applied, memory_order_relaxed);
}
