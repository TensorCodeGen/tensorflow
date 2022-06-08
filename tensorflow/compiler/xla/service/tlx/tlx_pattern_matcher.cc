#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/TensorType.h"

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"

#include "tensorflow/compiler/xla/service/tlx/tlx_pattern_matcher.h"

namespace xla {

    // Identifies whether the current HLO instruction 'Input' binds
    // to the root of a sub-graph which is equivalent to a Gelu activation
    GeluMatch* MatchGelu(HloInstruction* Input){

        LOG(INFO) << "Invoked MatchGeLu"<<"\n";
        /*
         * coeff = math_ops.cast(0.044715, features.dtype)
         0.5 * features * (
                1.0 + math_ops.tanh(0.7978845608028654 *
                                        (features + coeff * math_ops.pow(features, 3)))
         )*
         */


        HloInstruction* LeftMulOp;
        HloInstruction* RightMulOp;


        // lhs * rhs
        bool TopLevelMatch = Match(
                Input, 
                match::MultiplyAnyOrder(
                    match::Op(&LeftMulOp),
                    match::Op(&RightMulOp)
                    )
                );

        if(!TopLevelMatch)
            return nullptr;

        // One side will be calculating the (0.5 * 'features') while the other side
        // will be calculating (1 + tanh(...)). Here we check which correspond to
        // RightMulOp and LeftMulOp respectively.

        HloInstruction* Features;
        HloInstruction* TanH;

        auto TestTopLevelMatch = [&](HloInstruction* V1, HloInstruction* V2){

            bool TestLHS = Match(V1,
                    match::MultiplyAnyOrder(
                        match::Broadcast(match::ConstantScalar(0.5)),
                        match::Op(&Features)
                        )
                    );

            bool TestRHS = Match(V2,
                    match::AddAnyOrder(
                        match::Broadcast(match::ConstantScalar(1)),
                        match::Op(&TanH).WithOpcode(HloOpcode::kTanh)
                        )
                    );

            return TestLHS && TestRHS;

        };

        if(!TestTopLevelMatch(LeftMulOp, RightMulOp) && !TestTopLevelMatch(RightMulOp, LeftMulOp)){
            return nullptr;

        }

        // Features and TanH have been matched correctly, now we attempt
        // to match the operand to tanh with 
        // coeff = math_ops.cast(0.044715, features.dtype)
        // 0.797884583 * (features + coeff * math_ops.pow(features, 3))


        // Match the 0.797884583 * (...)
        HloInstruction* AddExpr;

        if(!Match((HloInstruction*) TanH->operand(0),
                    match::MultiplyAnyOrder(
                        match::Op(&AddExpr),
                        match::Broadcast(match::ConstantScalar(0.797884583))
                        )
                 )){
            return nullptr;
        }

        // AddExpr = 'features' + (coeff * math_ops.pow(features,3))

        HloInstruction* LeftAddend;
        HloInstruction* RightAddend;

        if(!Match(AddExpr,
                    match::AddAnyOrder(
                        match::Op(&LeftAddend),
                        match::Op(&RightAddend)
                        )
                 )){
            return nullptr;
        }



        HloInstruction* Features2;

        // (x Pow 3) * coeff is broken down into (x * x) * (x * coeff) by XLA,
        // so we match for (x * x) and x * coeff

        auto IsXMulCoeff = [&](HloInstruction* Op1){
            return Match(Op1,
                    match::MultiplyAnyOrder(
                        match::Broadcast(match::ConstantScalar(0.044715)),
                        match::Op(&Features2)
                        )
                    );
        };

        HloInstruction* SquareExprOp = nullptr;

        if(IsXMulCoeff(LeftAddend)){
            SquareExprOp = RightAddend;
        } else if(IsXMulCoeff(RightAddend)){
            SquareExprOp = LeftAddend;
        } else {
            return nullptr;
        }

        // Features2 is set to X, ensure that Features2 is the same as Features
        if(Features2 != Features){
            return nullptr;
        }


        HloInstruction* Features3;
        HloInstruction* Features4;

        if(!Match(SquareExprOp,
                    match::MultiplyAnyOrder(
                        match::Op(&Features3),
                        match::Op(&Features4)
                        )
                 )
          ){
            return nullptr;
        }

        // Verify that the same feature (i.e. input is being used for these expressions)
        // TODO: is there a way in the Match expression to state the feature must be the same
        // for multiple operands
        if(Features3 != Features4) return nullptr;
        if(Features3 != Features2) return nullptr;

        LOG(INFO) << "Matched Gelu operation" << "\n";

        GeluMatch* GM = new GeluMatch(Input, Features);

        return GM;

    }


}

