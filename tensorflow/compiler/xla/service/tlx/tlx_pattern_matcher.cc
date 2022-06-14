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

#include <cmath>        // std::abs

namespace xla {

    // Identifies whether the current HLO instruction 'Input' binds
    // to the root of a sub-graph which is equivalent to a Gelu activation
    GeluMatch* MatchGelu(HloInstruction* Input){

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
        HloInstruction* ScalarVal; 

        if(!Match((HloInstruction*) TanH->operand(0),
                    match::MultiplyAnyOrder(
                        match::Op(&AddExpr),
                        match::Broadcast(match::ConstantScalar(&ScalarVal))//0.79788453))
          )
            )){
                return nullptr;
            }

        // As the value 0.797 ... is ir-rational, using the exact value 
        // in the pattern may cause a match to be skipped. Hence we take 
        // the matched value and test whether it's absolute value is within
        // some epsilon delta from the specific value we require.

        auto double_equal = [](double v1, double v2) -> bool {
            double epsilon = 0.00001;
            LOG(INFO) << "Testing Double Equality between " << v1 << " and " << v2 << "\n";
            return std::abs(v1 - v2) < epsilon;
        };

        double scalar_val_test =  0.797884583;
        double matched_value = *ScalarVal->literal().GetAsDouble({});
        if(!double_equal(matched_value, scalar_val_test)){
            return nullptr;
        }


        // AddExpr = 'features' + (coeff * math_ops.pow(features,3))

        HloInstruction* FeaturesAddend;
        HloInstruction* RightAddend;

        if(!Match(AddExpr,
                    match::AddAnyOrder(
                        match::Op(&FeaturesAddend),
                        match::Op(&RightAddend).WithOpcode(HloOpcode::kMultiply)
                        )
                 )
          ){
            return nullptr;
        }



        // Match One of the addends to a arg type

        if(FeaturesAddend != Features){
            return nullptr;
        }



        LOG(INFO) << "REACHED LINED 151";

        // Right Addend is [(x Pow 3) * coeff]
        //
        // May be expressed in two possible tree structures

        auto MatchRightAddend1 = [&]()->  GeluMatch* {

            // (x Pow 3) * coeff is broken down into (x * x) * (x * coeff) by XLA,
            // so we match for (x * x) and x * coeff

            auto IsXMulCoeff = [&](HloInstruction* Op1){
                HloInstruction* ScalarMatch;
                HloInstruction* MatchFeatures;
                bool expr_match = Match(Op1,
                        match::MultiplyAnyOrder(
                            match::Broadcast(match::ConstantScalar(&ScalarMatch)),//0.044715)),
                        match::Op(&MatchFeatures)
                            )
                            );

                //expr_match = expr_match && (MatchFeatures == Features);


                if(expr_match){
                    double test_val = *ScalarMatch->literal().GetAsDouble({});
                    return double_equal(test_val, 0.044715);
                }

                return false;
            };

            HloInstruction* LeftProd;
            HloInstruction* RightProd;

            if(!Match(RightAddend,
                        match::MultiplyAnyOrder(
                            match::Op(&LeftProd),
                            match::Op(&RightProd)
                            )
                     )){
                return nullptr;
            }

            LOG(INFO) << "REACHED LINED 190";

            HloInstruction* SquareExprOp = nullptr;

            if(IsXMulCoeff(LeftProd)){
                SquareExprOp = RightProd;
            } else if(IsXMulCoeff(RightProd)){
                SquareExprOp = LeftProd;
            } else {
                return nullptr;
            }



            LOG(INFO) << "REACHED LINED 204";

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


            LOG(INFO) << "REACHED LINED 220";

            // Verify that the same feature (i.e. input is being used for these expressions)
            // TODO: is there a way in the Match expression to state the feature must be the same
            // for multiple operands
            if(Features3 != Features4) return nullptr;

            LOG(INFO) << "REACHED LINED 231";
            if(Features3 != Features) return nullptr;

            LOG(INFO) << "REACHED LINED 234";

            GeluMatch* GM = new GeluMatch(Input, Features);
            return GM;

        };


        auto MatchRightAddend2 = [&]()->  GeluMatch* {

            // (x Pow 3) * coeff is broken down into (((x*x) * 0.044715) * x)
            HloInstruction* OuterFeatureMatch;
            HloInstruction* InnerFeature1Match;
            HloInstruction* InnerFeature2Match;
            HloInstruction* ScalarMatch;


            LOG(INFO) << "Reached Line 250";

            if(!Match(RightAddend,
                        match::MultiplyAnyOrder(
                            match::MultiplyAnyOrder(
                                match::Broadcast(match::ConstantScalar(&ScalarMatch)),
                                match::MultiplyAnyOrder(
                                    match::Op(&InnerFeature1Match),
                                    match::Op(&InnerFeature2Match)
                                    )
                                ),
                            match::Op(&OuterFeatureMatch))
                     )
              ){
                return nullptr;
            }

            LOG(INFO) << "Reached Line 268";

            if(InnerFeature1Match != InnerFeature2Match) return nullptr;
            if(OuterFeatureMatch != InnerFeature2Match) return nullptr;
            if(OuterFeatureMatch != Features) return nullptr;

            double scalar_val_test =  0.044715;
            double matched_value = *ScalarMatch->literal().GetAsDouble({});
            if(!double_equal(matched_value, scalar_val_test)){
                return nullptr;
            }


            LOG(INFO) << "Reached Line 281";


            GeluMatch* GM = new GeluMatch(Input, Features);
            return GM;


        };

        // Explore both possible patterns
        GeluMatch* GM =  MatchRightAddend1();
        if(!GM)
            GM = MatchRightAddend2();

        return GM;

    }


}

