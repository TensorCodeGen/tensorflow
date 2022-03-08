

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/TensorType.h"


#include "tensorflow/compiler/xla/service/tlx/tlx_dot_op_emitter.h"





namespace xla {
namespace cpu {




void EmitTLXMatmul_Helper(const llvm_ir::IrArray& lhs_array_, const llvm_ir::IrArray& rhs_array_,
        const llvm_ir::IrArray& target_array_ ){

  const Shape& lhs_shape = lhs_array_.GetShape();
  const Shape& rhs_shape = rhs_array_.GetShape();


  llvm::Value* lhs = lhs_array_.GetBasePointer();
  llvm::Value* rhs = rhs_array_.GetBasePointer();
  llvm::Value* target = target_array_.GetBasePointer();

  llvm::LLVMContext & C = target->getContext();


  llvm::Type* ElemType = lhs_array_.GetElementLlvmType();

  llvm::Value* tlx_lhs_shape = GetShapeVector(lhs_shape, &C);
  llvm::Value* tlx_rhs_shape = GetShapeVector(rhs_shape, &C);


  llvm::Value* lhs_layout = GetLayoutVector(lhs_shape, &C);
  llvm::Value* rhs_layout = GetLayoutVector(rhs_shape, &C);


  // Create Empty padding vector
  llvm::Value* lhs_padding = Get0PaddingVector(lhs_shape, &C);
  llvm::Value* rhs_padding = Get0PaddingVector(rhs_shape, &C);
  
  // Creating the TLX Tensor types for the operands
  llvm::TensorType lhs_tensor_ty(tlx_lhs_shape, lhs_layout, lhs_padding);
  llvm::TensorType rhs_tensor_ty(tlx_rhs_shape, rhs_layout, rhs_padding);

    
}


}  // namespace llvm_ir
}  // namespace xla

