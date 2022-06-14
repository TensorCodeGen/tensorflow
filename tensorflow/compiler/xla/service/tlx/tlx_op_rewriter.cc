#include "tensorflow/compiler/xla/service/tlx/tlx_op_rewriter.h"


namespace xla {
    namespace cpu {

        bool TLXGeluRewriter::InstructionMatchesPattern(HloInstruction* instruction){

            GeluMatch* GM = MatchGelu(instruction);

            if(GM){
                LOG(INFO) << "[ TLX ] "<< "Found a match for the Gelu operation!"<<"\n";
                GeluInfo[instruction] = GM;
            }

            return false;
        }

        StatusOr<HloInstruction*> TLXGeluRewriter::ExpandInstruction(
                HloInstruction* instruction
                ){
            return nullptr;
        }
    }
}  // namespace xla
