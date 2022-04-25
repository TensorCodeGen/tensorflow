

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/TensorType.h"
#include "llvm/IR/Value.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/llvm_ir/kernel_support_library.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/service/tlx/tlx_utils.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace cpu {

void EmitTLXConv_Helper(const llvm_ir::IrArray& lhs_array_,
                        const llvm_ir::IrArray& rhs_array_,
                        const llvm_ir::IrArray& target_array_,
                        const int64 row_stride, const int64 col_stride,
                        const int64 lhs_row_dilation,
                        const int64 lhs_col_dilation, const int64 rhs_row_dilation,
                        const int64 rhs_col_dilation, llvm::IRBuilder<>* b_);

}  // namespace cpu
}  // namespace xla
