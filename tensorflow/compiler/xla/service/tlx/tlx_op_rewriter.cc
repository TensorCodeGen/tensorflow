#include "tensorflow/compiler/xla/service/tlx/tlx_op_rewriter.h"


namespace xla {
    namespace cpu {

        bool TLXGeluRewriter::InstructionMatchesPattern(HloInstruction* instruction){

            GeluMatch* GM = MatchGelu(instruction);

            if(GM){
                LOG(INFO) << "[ TLX ] "<< "Found a match for the Gelu operation!"<<"\n";
                GeluInfo[instruction] = GM;
            }

            return true;
        }

        StatusOr<HloInstruction*> TLXGeluRewriter::ExpandInstruction(
                HloInstruction* instruction
                ){
            if(GeluInfo.find(instruction) == GeluInfo.end()) return nullptr;

            GeluMatch* GM = GeluInfo[instruction];

            LOG(INFO) << "[ TLX ]" << " Replacing Gelu expression tree with Gelu call";

            HloInstruction* GeluInput = GM->GeluInput;


            auto* computation = instruction->parent();

            return computation->AddInstruction(
                HloInstruction::CreateCustomCall(/*shape=*/ GeluInput->shape(), /* operands=*/ {GeluInput}, /*target=*/ "tlx_gelu")
                    );


        }
    }
}  // namespace xla
