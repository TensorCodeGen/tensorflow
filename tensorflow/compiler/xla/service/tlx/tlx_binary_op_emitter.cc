#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/TensorType.h"


#include "tensorflow/compiler/xla/service/tlx/tlx_utils.h"
#include "tensorflow/compiler/xla/service/tlx/tlx_binary_op_emitter.h"


#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/llvm_ir/kernel_support_library.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/logging.h"



namespace xla {
    namespace cpu {

        bool TLXSupportsBinaryOp(HloInstruction* hlo){

            switch(hlo->opcode()){
                case HloOpcode::kAdd:
                case HloOpcode::kSubtract:
                case HloOpcode::kMultiply:
                case HloOpcode::kShiftLeft:
                case HloOpcode::kShiftRightArithmetic:
                case HloOpcode::kShiftRightLogical:
                    return true;
                default:
                    return false;
            }


        }


        void EmitTLXBinaryOp(HloInstruction* hlo, const llvm_ir::IrArray& lhs_array_, const llvm_ir::IrArray& rhs_array_,  const llvm_ir::IrArray& target_array_, llvm::IRBuilder<>* b_){


            LOG(INFO) << "[TLX]\t" << "EmitTLXBinaryOp"<<"\n";

            const Shape& lhs_shape = lhs_array_.GetShape();
            const Shape& rhs_shape = rhs_array_.GetShape();
            const Shape& target_shape = target_array_.GetShape();



            llvm::Value* lhs_ptr = lhs_array_.GetBasePointer();
            llvm::Value* rhs_ptr = rhs_array_.GetBasePointer();
            llvm::Value* target_ptr = target_array_.GetBasePointer();


            llvm::Type* LeftElemType = lhs_array_.GetElementLlvmType();
            llvm::Type* RightElemType = rhs_array_.GetElementLlvmType();
            llvm::Type* TargetElemType = target_array_.GetElementLlvmType();

            assert(LeftElemType && "Unspecified element type for lhs array");
            assert(RightElemType && "Unspecified element type for rhs array");
            assert(TargetElemType && "Unspecified element type for target array");


            llvm::LLVMContext & C = target_ptr->getContext();


            LOG(INFO) << "[TLX]\t" << "Get number of elements for the input tensors"<<"\n";
            int64_t num_lhs_values = GetNumElements(lhs_shape);
            int64_t num_rhs_values = GetNumElements(rhs_shape);
            int64_t num_target_values = GetNumElements(target_shape);;


            LOG(INFO) <<  "num_lhs_values:\t" << num_lhs_values <<"\n";
            LOG(INFO) <<  "num_rhs_values:\t" << num_rhs_values <<"\n";
            LOG(INFO) <<  "num_target_values:\t" << num_target_values <<"\n";


            assert(b_ && "IRBuilder pointer should be non-null");
            //auto InsertPoint = b_ -> saveIP();


            assert(lhs_ptr && "Expected a pointer for the lhs array");
            assert(llvm::isa<llvm::Instruction>(lhs_ptr) && "Expected the lhs pointer to be a LLVM  Instruction");


            LOG(INFO) << "Loading lhs vector";
            //b_ -> SetInsertPoint(llvm::dyn_cast<llvm::Instruction>(lhs_ptr) -> getNextNode());
            llvm::Value* lhs_vector = LoadPtrToVectorTy(lhs_ptr, LeftElemType, num_lhs_values, b_ );



            assert(rhs_ptr && "Expected a pointer for the rhs array");
            assert(llvm::isa<llvm::Instruction>(rhs_ptr) && "Expected the rhs pointer to be a LLVM  Instruction");

            LOG(INFO) << "Loading rhs vector";


            llvm::Value* rhs_vector = LoadPtrToVectorTy(rhs_ptr, RightElemType , num_rhs_values, b_ );



            LOG(INFO) << "[TLX]\t" << "Get Shape vectors "<<"\n";
            llvm::Value* tlx_lhs_shape = GetShapeVector(lhs_shape, &C);
            llvm::Value* tlx_rhs_shape = GetShapeVector(rhs_shape, &C);
            llvm::Value* tlx_target_shape = GetShapeVector(target_shape, &C);




            LOG(INFO) << "[TLX]\t" << "Get Layout vectors "<<"\n";
            llvm::Value* lhs_layout = GetLayoutVector(lhs_shape, &C);
            llvm::Value* rhs_layout = GetLayoutVector(rhs_shape, &C);
            llvm::Value* target_layout = GetLayoutVector(target_shape, &C);





            LOG(INFO) << "[TLX]\t" << "Get Padding vectors "<<"\n";
            // Create Empty padding vector
            llvm::Value* lhs_padding = Get0PaddingVector(lhs_shape, &C);
            llvm::Value* rhs_padding = Get0PaddingVector(rhs_shape, &C);
            llvm::Value* target_padding = Get0PaddingVector(target_shape, &C);


            //b_ -> SetInsertPoint(llvm::dyn_cast<llvm::Instruction>(lhs_vector) -> getNextNode());
            llvm::CallInst* lhs_type_info = CreateTypeInfoCall(lhs_vector, tlx_lhs_shape, lhs_layout, lhs_padding, b_);


            //b_ -> SetInsertPoint(llvm::dyn_cast<llvm::Instruction>(rhs_vector) -> getNextNode());
            llvm::CallInst* rhs_type_info = CreateTypeInfoCall(rhs_vector, tlx_rhs_shape, rhs_layout, rhs_padding, b_);



            //b_ ->restoreIP(InsertPoint);

            llvm::Value* Result = nullptr;

            switch(hlo->opcode()){
                case HloOpcode::kAdd:
                    if(TargetElemType->isFloatTy()){
                        Result = b_->CreateFAdd(lhs_vector, rhs_vector, "tensor_fadd");
                    } else {
                        Result = b_->CreateAdd(lhs_vector, rhs_vector, "tensor_iadd");
                    }
                    break;
                case HloOpcode::kSubtract:
                    if(TargetElemType->isFloatTy()){
                        Result = b_->CreateFSub(lhs_vector, rhs_vector, "tensor_fsub");
                    } else {
                        Result = b_->CreateSub(lhs_vector, rhs_vector, "tensor_isub");
                    }
                    break;
                case HloOpcode::kMultiply:
                    if(TargetElemType->isFloatTy()){
                        Result = b_->CreateFMul(lhs_vector, rhs_vector, "tensor_fmul");
                    } else {
                        Result = b_->CreateMul(lhs_vector, rhs_vector, "tensor_imul");
                    }
                    break;
                case HloOpcode::kShiftLeft:
                    Result = b_->CreateShl(lhs_vector, rhs_vector, "tensor_shl");
                    break;
                case HloOpcode::kShiftRightArithmetic:
                    Result = b_->CreateAShr(lhs_vector, rhs_vector, "tensor_ashr");
                    break;
                case HloOpcode::kShiftRightLogical:
                    Result = b_->CreateLShr(lhs_vector, rhs_vector, "tensor_lshr");
                    break;
                default:
                    assert(false && "Unsupported binary op");
                    break;
            }

            llvm::CallInst* target_type_info = CreateTypeInfoCall(Result, tlx_target_shape, target_layout, target_padding, b_);



            LOG(INFO) << "[TLX]\t" << "Create store back for result"<<"\n";


            // To support those operations not supported by TLX
            // we store the output back into the target IR Array so 
            // XLA can use this output.
            llvm::StoreInst* StoreResult = StoreVectorTyToPtr(Result, target_ptr, TargetElemType, num_target_values , b_ );



            LOG(INFO) << "[TLX]\t" << "Completed generation of TLX Binary Op "<<"\n";

        }


    }  // namespace llvm_ir
}  // namespace xla

