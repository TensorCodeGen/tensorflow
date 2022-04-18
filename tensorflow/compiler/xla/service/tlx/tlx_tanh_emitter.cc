

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/TensorType.h"


#include "tensorflow/compiler/xla/service/tlx/tlx_tanh_emitter.h"
#include "tensorflow/core/platform/logging.h"




namespace xla {
    namespace cpu {


        llvm::CallInst* CreateTanhCall(llvm::CallInst* SourceTypeInfo, llvm::VectorType* TargetVecTy, llvm::IRBuilder<>* b_){

            llvm::Module* M= b_->GetInsertBlock()->getParent()->getParent();

            llvm::LLVMContext& Ctx = M->getContext();

            std::vector<llvm::Type*> TanhArgsTy = {TargetVecTy};

            llvm::Function* TanhFn = llvm::Intrinsic::getDeclaration(M, llvm::Intrinsic::tensor_tanh, llvm::ArrayRef<llvm::Type*>(TanhArgsTy) );
            std::vector<llvm::Value*> TanhArgs = {SourceTypeInfo};

            llvm::CallInst* CI = b_->CreateCall(TanhFn->getFunctionType(), TanhFn, llvm::ArrayRef<llvm::Value*>(TanhArgs), "llvm_tanh");

            return CI;

        }


        void EmitTLXTanh_Helper(const llvm_ir::IrArray& source_array_, const llvm_ir::IrArray& target_array_, llvm::IRBuilder<>* b_){


            LOG(INFO) << "[TLX]\t" << "Emit TLX Tanh Helper"<<"\n";


            llvm::Module* M= b_->GetInsertBlock()->getParent()->getParent();

            llvm::LLVMContext& C = M->getContext();


            const Shape& source_shape = source_array_.GetShape();
            const Shape& target_shape = target_array_.GetShape();


            llvm::Type* SourceElemType = source_array_.GetElementLlvmType();
            llvm::Type* TargetElemType = target_array_.GetElementLlvmType();

            int64_t num_source_values = GetNumElements(source_shape);
            int64_t num_target_values = GetNumElements(target_shape);;


            auto InsertPoint = b_ -> saveIP();


            llvm::Value* source_ptr = source_array_.GetBasePointer();
            llvm::Value* target_ptr = target_array_.GetBasePointer();


            b_ -> SetInsertPoint(llvm::dyn_cast<llvm::Instruction>(source_ptr) -> getNextNode());
            llvm::Value* source_vector = LoadPtrToVectorTy(source_ptr, SourceElemType, num_source_values, b_ );


            llvm::Value* tlx_source_shape = GetShapeVector(source_shape, &C);


            llvm::Value* tlx_source_layout = GetLayoutVector(source_shape, &C);


            llvm::Value* tlx_source_padding = Get0PaddingVector(source_shape, &C);


            b_ -> SetInsertPoint(llvm::dyn_cast<llvm::Instruction>(source_vector) -> getNextNode());
            llvm::CallInst* source_type_info = CreateTypeInfoCall(source_vector, tlx_source_shape, tlx_source_layout, tlx_source_padding, b_);


            b_ ->restoreIP(InsertPoint);


            llvm::VectorType*  TargetVecTy = llvm::FixedVectorType::get(TargetElemType, num_target_values);

            llvm::CallInst* Tanh_vector = CreateTanhCall(source_type_info, TargetVecTy, b_);


            LOG(INFO) << "[TLX]\t" << "Create tensor tanh typeinfo call"<<"\n";

            llvm::CallInst* target_type_info = CreateTypeInfoCall(Tanh_vector, tlx_source_shape, tlx_source_layout, tlx_source_padding, b_);


            LOG(INFO) << "[TLX]\t" << "Create store back for result"<<"\n";


            llvm::StoreInst* StoreResult = StoreVectorTyToPtr(Tanh_vector, target_ptr, TargetElemType, num_target_values , b_ );



            LOG(INFO) << "[TLX]\t" << "Completed generation of TLX Tanh "<<"\n";

        }


    }  // namespace llvm_ir
}  // namespace xla

