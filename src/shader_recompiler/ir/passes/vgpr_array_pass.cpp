// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <unordered_map>
#include <unordered_set>
#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/basic_block.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/program.h"

namespace Shader::Optimization {

static std::unordered_set<u32> CollectLaneRegisters(const IR::Program& program) {
    std::unordered_set<u32> lane_regs;
    for (IR::Block* const block : program.blocks) {
        for (const IR::Inst& inst : block->Instructions()) {
            const auto op = inst.GetOpcode();
            if (op == IR::Opcode::SetVectorRegister && !inst.Arg(1).IsImmediate() &&
                inst.Arg(1).Inst()->GetOpcode() == IR::Opcode::WriteLane) {
                lane_regs.insert(static_cast<u32>(inst.Arg(0).VectorReg()));
            } else if ((op == IR::Opcode::ReadLane || op == IR::Opcode::WriteLane) &&
                       !inst.Arg(0).IsImmediate()) {
                const IR::Inst* src = inst.Arg(0).Inst();
                if (src->GetOpcode() == IR::Opcode::GetVectorRegister) {
                    lane_regs.insert(static_cast<u32>(src->Arg(0).VectorReg()));
                }
            }
        }
    }
    return lane_regs;
}

void VgprArrayPass(IR::Program& program) {
    const u32 base = program.info.vgpr_array_base;
    if (base == ~0u) {
        return;
    }
    // ReadLaneEliminationPass needs the WriteLane and ReadLane of a register connected through SSA.
    const auto lane_regs = CollectLaneRegisters(program);
    for (IR::Block* const block : program.blocks) {
        for (auto it = block->Instructions().begin(); it != block->Instructions().end(); ++it) {
            IR::Inst& inst = *it;
            const auto op = inst.GetOpcode();
            if (op != IR::Opcode::GetVectorRegister && op != IR::Opcode::SetVectorRegister) {
                continue;
            }
            const u32 reg = static_cast<u32>(inst.Arg(0).VectorReg());
            if (reg < base || lane_regs.contains(reg)) {
                continue;
            }
            IR::IREmitter ir{*block, it};
            if (op == IR::Opcode::GetVectorRegister) {
                inst.ReplaceUsesWith(ir.LoadVgprArray(ir.Imm32(reg)));
            } else {
                ir.StoreVgprArray(ir.Imm32(reg), IR::U32{inst.Arg(1)});
                inst.Invalidate();
            }
        }
        std::unordered_map<u32, IR::Value> known;
        for (IR::Inst& inst : block->Instructions()) {
            if (inst.GetOpcode() == IR::Opcode::StoreVgprArray) {
                if (inst.Arg(0).IsImmediate()) {
                    known[inst.Arg(0).U32()] = inst.Arg(1);
                } else {
                    known.clear();
                }
            } else if (inst.GetOpcode() == IR::Opcode::LoadVgprArray && inst.Arg(0).IsImmediate()) {
                if (const auto found = known.find(inst.Arg(0).U32()); found != known.end()) {
                    inst.ReplaceUsesWith(found->second);
                }
            }
        }
    }
}

} // namespace Shader::Optimization
