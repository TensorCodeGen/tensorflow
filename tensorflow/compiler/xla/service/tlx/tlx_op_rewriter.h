#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/TensorType.h"


#include "tensorflow/compiler/xla/service/tlx/tlx_utils.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/service/op_expander_pass.h"
#include "tensorflow/compiler/xla/service/tlx/tlx_pattern_matcher.h"

#include <map>

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_TLX_OP_REWRITER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_TLX_OP_REWRITER_H_



namespace xla {

    namespace cpu {


        class TLXGeluRewriter : public OpExpanderPass {
            public:
                explicit TLXGeluRewriter(PatternExtraFilter extra_filter = nullptr)
                    : OpExpanderPass(std::move(extra_filter)) {}
                absl::string_view name() const override { return "tlx_rewriter"; }


                std::map<HloInstruction*, GeluMatch*> GeluInfo;

                virtual ~TLXGeluRewriter() {}
            protected:
                  bool InstructionMatchesPattern(HloInstruction* instruction) override;
                  StatusOr<HloInstruction*> ExpandInstruction(HloInstruction* instruction) override;


        };

    } 
}  // namespace xla

#endif
