

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/TensorType.h"


#include "tensorflow/compiler/xla/service/tlx/tlx_map_emitter.h"
#include "tensorflow/core/platform/logging.h"




namespace xla {
    namespace cpu {

        llvm::CallInst* CreateMapCall(llvm::CallInst* SourceTypeInfo, llvm::Function* ElementalFunction , llvm::VectorType* TargetVecTy, llvm::IRBuilder<>* b_){

            LOG(INFO) << "[TLX]\t" << "Create tensor map call"<<"\n";

            llvm::Module* M= b_->GetInsertBlock()->getParent()->getParent();

            llvm::LLVMContext& Ctx = M->getContext();
            llvm::Type* I8PtrTy = llvm::Type::getInt8PtrTy(Ctx);

            llvm::Value* BitcastFunc  = b_->CreateBitCast(ElementalFunction, I8PtrTy);
            std::vector<llvm::Value*> MapArgs = {SourceTypeInfo, BitcastFunc};
            std::vector<llvm::Type*> MapArgsTy = {TargetVecTy, BitcastFunc->getType()};


            llvm::Function* TensorMapFn = llvm::Intrinsic::getDeclaration(M, llvm::Intrinsic::tensor_map, llvm::ArrayRef<llvm::Type*>(MapArgsTy) );

            llvm::CallInst* CI = b_->CreateCall(TensorMapFn->getFunctionType(), TensorMapFn, llvm::ArrayRef<llvm::Value*>(MapArgs), "llvm_map");

            return CI;

        }


    }  // namespace llvm_ir
}  // namespace xla

