

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/TensorType.h"


#include "tensorflow/compiler/xla/service/tlx/tlx_dot_op_emitter.h"
#include "tensorflow/core/platform/logging.h"




namespace xla {
namespace cpu {




void EmitTLXMatmul_Helper(const llvm_ir::IrArray& lhs_array_, const llvm_ir::IrArray& rhs_array_,
        const llvm_ir::IrArray& target_array_ , llvm::IRBuilder<>* b_){

  LOG(INFO) << "[TLX]\t" << "EmitTLXMatmul_Helper"<<"\n";
  const Shape& lhs_shape = lhs_array_.GetShape();
  const Shape& rhs_shape = rhs_array_.GetShape();
  const Shape& target_shape = target_array_.GetShape();



  llvm::Value* lhs_ptr = lhs_array_.GetBasePointer();
  llvm::Value* rhs_ptr = rhs_array_.GetBasePointer();
  llvm::Value* target_ptr = target_array_.GetBasePointer();


  llvm::Type* LeftElemType = lhs_array_.GetElementLlvmType();
  llvm::Type* RightElemType = rhs_array_.GetElementLlvmType();
  llvm::Type* TargetElemType = target_array_.GetElementLlvmType();


  llvm::LLVMContext & C = target_ptr->getContext();


  LOG(INFO) << "[TLX]\t" << "Get number of elements for the input tensors"<<"\n";
  int64_t num_lhs_values = GetNumElements(lhs_shape);
  int64_t num_rhs_values = GetNumElements(rhs_shape);
  int64_t num_target_values = GetNumElements(target_shape);;


  LOG(INFO) <<  "num_lhs_values:\t" << num_lhs_values <<"\n";
  LOG(INFO) <<  "num_rhs_values:\t" << num_rhs_values <<"\n";
  LOG(INFO) <<  "num_target_values:\t" << num_target_values <<"\n";



  LOG(INFO) << "[TLX]\t" << "Load Pointers into llvm vector type "<<"\n";

  auto InsertPoint = b_ -> saveIP();



  b_ -> SetInsertPoint(llvm::dyn_cast<llvm::Instruction>(lhs_ptr) -> getNextNode());
  llvm::Value* lhs_vector = LoadPtrToVectorTy(lhs_ptr, LeftElemType, num_lhs_values, b_ );

  b_ -> SetInsertPoint(llvm::dyn_cast<llvm::Instruction>(rhs_ptr) -> getNextNode());
  llvm::Value* rhs_vector = LoadPtrToVectorTy(rhs_ptr, RightElemType , num_rhs_values, b_ );
  



  LOG(INFO) << "[TLX]\t" << "Get Shape vectors "<<"\n";
  llvm::Value* tlx_lhs_shape = GetShapeVector(lhs_shape, &C);
  llvm::Value* tlx_rhs_shape = GetShapeVector(rhs_shape, &C);
  llvm::Value* tlx_target_shape = GetShapeVector(target_shape, &C);

  //LOG(INFO) << "lhs shape "<< *tlx_lhs_shape <<"\n";
  //LOG(INFO) << "rhs shape "<< *tlx_rhs_shape <<"\n";
  //LOG(INFO) << "target shape "<< *tlx_target_shape <<"\n";



  LOG(INFO) << "[TLX]\t" << "Get Layout vectors "<<"\n";
  llvm::Value* lhs_layout = GetLayoutVector(lhs_shape, &C);
  llvm::Value* rhs_layout = GetLayoutVector(rhs_shape, &C);
  llvm::Value* target_layout = GetLayoutVector(target_shape, &C);





  LOG(INFO) << "[TLX]\t" << "Get Padding vectors "<<"\n";
  // Create Empty padding vector
  llvm::Value* lhs_padding = Get0PaddingVector(lhs_shape, &C);
  llvm::Value* rhs_padding = Get0PaddingVector(rhs_shape, &C);
  llvm::Value* target_padding = Get0PaddingVector(target_shape, &C);


  LOG(INFO) << "[TLX]\t" << "Create tensor types for operands "<<"\n";
  
  // Creating the TLX Tensor types for the operands
  llvm::TensorType lhs_tensor_ty(tlx_lhs_shape, lhs_layout, lhs_padding);
  llvm::TensorType rhs_tensor_ty(tlx_rhs_shape, rhs_layout, rhs_padding);


  LOG(INFO) << "[TLX]\t" << "Create tensor typeinfo for operands "<<"\n";


  b_ -> SetInsertPoint(llvm::dyn_cast<llvm::Instruction>(lhs_vector) -> getNextNode());
  // Create typeinfo calls for tensor operands
  llvm::CallInst* lhs_type_info = CreateTypeInfoCall(lhs_vector, tlx_lhs_shape, lhs_layout, lhs_padding, b_);


  b_ -> SetInsertPoint(llvm::dyn_cast<llvm::Instruction>(rhs_vector) -> getNextNode());
  llvm::CallInst* rhs_type_info = CreateTypeInfoCall(rhs_vector, tlx_rhs_shape, rhs_layout, rhs_padding, b_);


  LOG(INFO) << "[TLX]\t" << "Create tensor matmul "<<"\n";

  b_ ->restoreIP(InsertPoint);
  llvm::VectorType*  TargetVecTy = llvm::FixedVectorType::get(TargetElemType, num_target_values);
  // Create Tensor Matmul call
  llvm::CallInst* Matmul_vector = CreateMatMulCall(lhs_type_info, rhs_type_info, TargetVecTy ,  b_);

  LOG(INFO) << "[TLX]\t" << "Create tensor matmul typeinfo call"<<"\n";
  llvm::CallInst* target_type_info = CreateTypeInfoCall(Matmul_vector, tlx_target_shape, target_layout, target_padding, b_);
  


  LOG(INFO) << "[TLX]\t" << "Create store back for result"<<"\n";


  // To support those operations not supported by TLX
  // we store the output back into the target IR Array so 
  // XLA can use this output.
  llvm::StoreInst* StoreResult = StoreVectorTyToPtr(Matmul_vector, target_ptr, TargetElemType, num_target_values , b_ );


  LOG(INFO) << "[TLX]\t" << "Completed generation of TLX Dot "<<"\n";

  return;

    
}


}  // namespace llvm_ir
}  // namespace xla

