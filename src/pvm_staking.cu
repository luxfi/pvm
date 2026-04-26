// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// pvm_staking.cu — CUDA peer of pvm_staking.metal.

#include "pvm_kernels_common.cuh"

namespace pvm::cuda {

extern "C" __global__ void pvm_stake_transition(
    const PVMRoundDescriptor* desc,
    const StakeOp*            ops,
    ValidatorSlot*            validators,
    StakeRecord*              stake,
    uint32_t*                 applied_out,
    uint32_t                  validator_count,
    uint32_t                  stake_count)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    uint32_t applied = 0;
    uint32_t count = desc->stake_op_count;
    for (uint32_t i = 0; i < count; ++i) {
        const StakeOp& op = ops[i];
        switch (op.kind) {
            case kSOpBond: {
                uint32_t v_idx = validator_locate(validators, validator_count,
                                                  op.validator_id, false);
                if (v_idx == 0xFFFFFFFFu) break;
                ValidatorSlot& v = validators[v_idx];
                if ((v.status & kStatusTombstoned) != 0u) break;

                uint32_t s_idx = stake_record_locate(stake, stake_count,
                                                    op.delegator_id,
                                                    op.validator_id, true);
                if (s_idx == 0xFFFFFFFFu) break;
                StakeRecord& s = stake[s_idx];
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
                uint32_t s_idx = stake_record_locate(stake, stake_count,
                                                    op.delegator_id,
                                                    op.validator_id, false);
                if (s_idx == 0xFFFFFFFFu) break;
                StakeRecord& s = stake[s_idx];
                if (s.status != kStakeStatusActive) break;
                if (op.epoch < s.lock_until_epoch) break;
                uint64_t amt = (op.amount < s.amount) ? op.amount : s.amount;
                s.amount = sat_sub_u64(s.amount, amt);
                s.epoch_unbonded = op.epoch;
                s.status = (s.amount == 0) ? kStakeStatusRetired : kStakeStatusUnbonding;

                uint32_t v_idx = validator_locate(validators, validator_count,
                                                  op.validator_id, false);
                if (v_idx != 0xFFFFFFFFu) {
                    ValidatorSlot& v = validators[v_idx];
                    v.weight = sat_sub_u64(v.weight, amt);
                }
                ++applied;
                break;
            }
            case kSOpDelegate: {
                uint32_t v_idx = validator_locate(validators, validator_count,
                                                  op.validator_id, false);
                if (v_idx == 0xFFFFFFFFu) break;
                ValidatorSlot& v = validators[v_idx];
                if ((v.status & kStatusTombstoned) != 0u) break;
                if ((v.status & kStatusJailed) != 0u) break;

                uint32_t s_idx = stake_record_locate(stake, stake_count,
                                                    op.delegator_id,
                                                    op.validator_id, true);
                if (s_idx == 0xFFFFFFFFu) break;
                StakeRecord& s = stake[s_idx];
                s.amount = sat_add_u64(s.amount, op.amount);
                s.status = kStakeStatusActive;
                if (s.epoch_bonded == 0) s.epoch_bonded = op.epoch;
                v.weight = sat_add_u64(v.weight, op.amount);
                ++applied;
                break;
            }
            case kSOpRedelegate: {
                if (op.source_validator_id == op.validator_id) break;

                uint32_t src_idx = stake_record_locate(stake, stake_count,
                                                      op.delegator_id,
                                                      op.source_validator_id, false);
                if (src_idx == 0xFFFFFFFFu) break;
                StakeRecord& src = stake[src_idx];
                if (src.status != kStakeStatusActive) break;
                if (op.epoch < src.lock_until_epoch) break;

                uint32_t v_dst_idx = validator_locate(validators, validator_count,
                                                     op.validator_id, false);
                if (v_dst_idx == 0xFFFFFFFFu) break;
                ValidatorSlot& v_dst = validators[v_dst_idx];
                if ((v_dst.status & kStatusTombstoned) != 0u) break;

                uint64_t amt = (op.amount < src.amount) ? op.amount : src.amount;
                src.amount = sat_sub_u64(src.amount, amt);
                if (src.amount == 0) src.status = kStakeStatusRetired;

                uint32_t v_src_idx = validator_locate(validators, validator_count,
                                                     op.source_validator_id, false);
                if (v_src_idx != 0xFFFFFFFFu) {
                    ValidatorSlot& v_src = validators[v_src_idx];
                    v_src.weight = sat_sub_u64(v_src.weight, amt);
                }

                uint32_t dst_idx = stake_record_locate(stake, stake_count,
                                                      op.delegator_id,
                                                      op.validator_id, true);
                if (dst_idx == 0xFFFFFFFFu) break;
                StakeRecord& dst = stake[dst_idx];
                dst.amount = sat_add_u64(dst.amount, amt);
                dst.status = kStakeStatusActive;
                if (dst.epoch_bonded == 0) dst.epoch_bonded = op.epoch;
                v_dst.weight = sat_add_u64(v_dst.weight, amt);
                ++applied;
                break;
            }
            case kSOpReward: {
                uint32_t v_idx = validator_locate(validators, validator_count,
                                                  op.validator_id, false);
                if (v_idx == 0xFFFFFFFFu) break;
                ValidatorSlot& v = validators[v_idx];
                if (v.weight == 0) break;
                uint64_t scaled = (op.amount > UINT64_MAX / kRewardScale)
                    ? UINT64_MAX
                    : op.amount * kRewardScale;
                uint64_t per_unit = scaled / v.weight;
                if (per_unit == 0) break;
                for (uint32_t si = 0; si < stake_count; ++si) {
                    StakeRecord& s = stake[si];
                    if (s.status != kStakeStatusActive) continue;
                    if (s.validator_id != op.validator_id) continue;
                    uint64_t delta = (s.amount > UINT64_MAX / per_unit)
                        ? UINT64_MAX
                        : s.amount * per_unit;
                    s.reward_accumulator = sat_add_u64(s.reward_accumulator, delta);
                }
                ++applied;
                break;
            }
            case kSOpCommission: {
                uint32_t v_idx = validator_locate(validators, validator_count,
                                                  op.validator_id, false);
                if (v_idx == 0xFFFFFFFFu) break;
                if (op.commission_bps > 10000u) break;
                uint32_t s_idx = stake_record_locate(stake, stake_count,
                                                    op.validator_id,
                                                    op.validator_id, false);
                if (s_idx == 0xFFFFFFFFu) break;
                stake[s_idx].commission_bps = op.commission_bps;
                ++applied;
                break;
            }
            default: break;
        }
    }
    *applied_out = applied;
}

}  // namespace pvm::cuda
