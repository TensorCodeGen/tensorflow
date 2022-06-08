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

#include "tensorflow/compiler/xla/service/tlx/tlx_utils.h"

namespace xla {

    struct GeluMatch {
        HloInstruction* GeluRoot; // The Root node of the HloExpression graph which returns the GeLu output
        HloInstruction* GeluInput; // The input feature passed as input to the Gelu calculation

        GeluMatch(HloInstruction* GR, HloInstruction* GI) : GeluRoot(GR), GeluInput(GI) {};
    };

    GeluMatch* MatchGelu(HloInstruction* Input);


}

