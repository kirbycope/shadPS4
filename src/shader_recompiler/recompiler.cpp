// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <unordered_map>
#include <unordered_set>
#include "common/logging/classes.h"
#include "shader_recompiler/frontend/control_flow_graph.h"
#include "shader_recompiler/frontend/decode.h"
#include "shader_recompiler/frontend/structured_control_flow.h"
#include "shader_recompiler/frontend/translate/translate.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/post_order.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"

namespace Shader {

// VGPRs at or above info.vgpr_array_base are reachable by relative moves, so keep them in the
// per-invocation VGPR array instead of SSA: rewrite their plain reads and writes into array
// accesses before SSA construction. Runs only for shaders that contain relative moves.
static void VgprArrayPass(IR::Program& program, const Info& info) {
    if (info.vgpr_array_base == ~0u) {
        return;
    }
    // Registers used as per-lane scratch through v_writelane/v_readlane (games save EXEC there)
    // stay plain registers: ReadLaneEliminationPass can only fold a ReadLane into the value its
    // WriteLane stored when the two are connected through SSA, and a round trip through the array
    // would leave a real cross-lane read of lanes that may not be active.
    std::unordered_set<u32> lane_regs;
    for (IR::Block* const block : program.blocks) {
        for (IR::Inst& inst : block->Instructions()) {
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
    for (IR::Block* const block : program.blocks) {
        for (auto it = block->Instructions().begin(); it != block->Instructions().end(); ++it) {
            IR::Inst& inst = *it;
            const auto op = inst.GetOpcode();
            if (op != IR::Opcode::GetVectorRegister && op != IR::Opcode::SetVectorRegister) {
                continue;
            }
            const u32 reg = static_cast<u32>(inst.Arg(0).VectorReg());
            if (reg < info.vgpr_array_base || lane_regs.contains(reg)) {
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
        // Forward stores to later constant-index loads of the same slot within the block, until a
        // dynamically indexed store could have overwritten it. This keeps values that are produced
        // and consumed nearby (such as resource sharps) visible to later passes as plain SSA.
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

IR::BlockList GenerateBlocks(const IR::AbstractSyntaxList& syntax_list) {
    size_t num_syntax_blocks{};
    for (const auto& [_, type] : syntax_list) {
        if (type == IR::AbstractSyntaxNode::Type::Block) {
            ++num_syntax_blocks;
        }
    }
    IR::BlockList blocks{};
    blocks.reserve(num_syntax_blocks);
    for (const auto& [data, type] : syntax_list) {
        if (type == IR::AbstractSyntaxNode::Type::Block) {
            blocks.push_back(data.block);
        }
    }
    return blocks;
}

void EmitControlFlowGraph(IR::Program& program, Pools& pools, Gcn::CFG& cfg,
                          RuntimeInfo& runtime_info, const Profile& profile) {
    Gcn::Translator translator{program.info, runtime_info, profile};
    for (auto& block : cfg) {
        const u32 start = block.begin_index;
        const u32 size = block.end_index - start + 1;
        auto* ir_block = pools.block_pool.Create(pools.inst_pool);
        ir_block->cfg_block = &block;
        block.ir_block = ir_block;
        translator.Translate(ir_block, block.begin, block.cond,
                             std::span{program.ins_list}.subspan(start, size));
        program.blocks.push_back(ir_block);
    }
    translator.EmitPrologue(program.blocks.front());
    ASSERT_MSG(!program.info.translation_failed, "Shader translation has failed");
    for (auto& block : cfg) {
        auto* ir_block = block.ir_block;
        if (block.branch_true) {
            auto* true_block = block.branch_true->ir_block;
            ir_block->AddBranch(true_block);
        }
        if (block.branch_false) {
            auto* false_block = block.branch_false->ir_block;
            ir_block->AddBranch(false_block);
        }
    }
    program.post_order_blocks = Shader::IR::PostOrder(program.blocks.front());
}

IR::Program TranslateProgram(const std::span<const u32>& code, Pools& pools, Info& info,
                             RuntimeInfo& runtime_info, const Profile& profile) {
    // Ensure first instruction is expected.
    constexpr u32 token_mov_vcchi = 0xBEEB03FF;
    if (code[0] != token_mov_vcchi) {
        LOG_WARNING(Render_Recompiler, "First instruction is not s_mov_b32 vcc_hi, #imm");
    }

    Gcn::GcnCodeSlice slice(code.data(), code.data() + code.size());
    Gcn::GcnDecodeContext decoder;

    // Decode and save instructions
    IR::Program program{info};
    program.ins_list.reserve(code.size());
    while (!slice.atEnd()) {
        program.ins_list.emplace_back(decoder.decodeInstruction(slice));
    }

    // Clear any previous pooled data.
    pools.ReleaseContents();

    // Create control flow graph
    Common::ObjectPool<Gcn::Block> gcn_block_pool{64};
    Gcn::CFG cfg{gcn_block_pool, program.ins_list};
    EmitControlFlowGraph(program, pools, cfg, runtime_info, profile);

    // On NVIDIA GPUs HW interpolation of clip distance values seems broken, and we need to emulate
    // it with expensive discard in PS.
    Shader::InjectClipDistanceAttributes(program, runtime_info);

    // Move every VGPR a relative move can reach into the per-invocation array.
    VgprArrayPass(program, info);

    // Run optimization passes on unstructured graph
    if (!profile.support_float64) {
        Shader::Optimization::LowerFp64ToFp32(program);
    }
    Shader::Optimization::SsaRewritePass(program);
    Shader::Optimization::ConstantPropagationPass(program.post_order_blocks);
    Shader::Optimization::ReadLaneEliminationPass(program);
    if (info.sw_stage == SwStage::TessellationControl) {
        Shader::Optimization::TessellationPreprocess(program, runtime_info);
        Shader::Optimization::HullShaderTransform(program, runtime_info);
    } else if (info.sw_stage == SwStage::TessellationEval) {
        Shader::Optimization::TessellationPreprocess(program, runtime_info);
        Shader::Optimization::DomainShaderTransform(program, runtime_info);
    }
    Shader::Optimization::RingAccessElimination(program, runtime_info);
    Shader::IR::DumpProgram(program, info, "pre-res-discover.");
    auto resources = Shader::Optimization::ResourceDiscoverPass(program, profile);
    Shader::Optimization::FlattenExtendedUserdataPass(program);
    Shader::IR::DumpProgram(program, info, "pre-res-patch.");
    Shader::Optimization::ResourcePatchingPass(program.info, resources, profile);
    Shader::Optimization::LowerBufferFormatToRaw(program);
    Shader::Optimization::SharedMemorySimplifyPass(program, profile);
    Shader::Optimization::SharedMemoryToStoragePass(program, runtime_info, profile);
    Shader::Optimization::LowerUserClipPlanes(program, runtime_info);
    Shader::Optimization::PhiSimplificationPass(program);
    Shader::Optimization::InverseBallotEliminationPass(program);
    Shader::IR::DumpProgram(program, info, "pre-lower-phi.");

    // Prepare for structurization by clearing flow graph and lowering phis
    for (auto* ir_block : program.blocks) {
        ir_block->imm_predecessors.clear();
        ir_block->imm_successors.clear();
        ir_block->ssa_state.Reset();
    }
    Shader::Optimization::LowerPhisToRegsPass(program);

    // Structurize control flow graph and create program.
    program.syntax_list = Shader::Gcn::BuildASL(pools, cfg, info);
    program.blocks = GenerateBlocks(program.syntax_list);
    program.post_order_blocks = Shader::IR::PostOrder(program.syntax_list.front().data.block);

    // Run optimization passes on structured graph
    Shader::Optimization::SsaRepairPass(program);
    Shader::Optimization::SsaRewritePass(program);
    Shader::Optimization::SharedMemoryBarrierPass(program, runtime_info, profile);
    Shader::Optimization::DeadCodeEliminationPass(program);
    Shader::Optimization::LowerWave64BallotPass(program, runtime_info, profile);
    Shader::Optimization::LowerHardwareIntrinsics(program);
    Shader::Optimization::ConstantPropagationPass(program.post_order_blocks);
    Shader::Optimization::DeadCodeEliminationPass(program);
    Shader::Optimization::CollectShaderInfoPass(program, profile);
    Shader::IR::DumpProgram(program, info);

    return program;
}

} // namespace Shader
