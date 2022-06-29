

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

        llvm::CallInst* CreateGeneralMapCall(std::vector<llvm::CallInst*> SourceTypeInfos, llvm::Function* ElementalFunction , llvm::VectorType* TargetVecTy, llvm::IRBuilder<>* b_){

            LOG(INFO) << "[TLX]\t" << "Create general tensor map call"<<"\n";

            llvm::Module* M= b_->GetInsertBlock()->getParent()->getParent();

            llvm::LLVMContext& Ctx = M->getContext();
            llvm::Type* I8PtrTy = llvm::Type::getInt8PtrTy(Ctx);
            llvm::Type* I32Ty = llvm::Type::getInt32Ty(Ctx);
            llvm::Constant* N = llvm::ConstantInt::get(I32Ty, SourceTypeInfos.size());

            llvm::Value* BitcastFunc  = b_->CreateBitCast(ElementalFunction, I8PtrTy);
            std::vector<llvm::Value*> MapArgs = {BitcastFunc, N};
            MapArgs.insert(MapArgs.end(), SourceTypeInfos.begin(), SourceTypeInfos.end());

            std::vector<llvm::Type*> MapArgsTy = {TargetVecTy, BitcastFunc->getType()};

            for(auto* ST : SourceTypeInfos){
                MapArgsTy.push_back(ST->getType());
            }


            llvm::Function* TensorMapFn = llvm::Intrinsic::getDeclaration(M, llvm::Intrinsic::general_tensor_map, llvm::ArrayRef<llvm::Type*>(MapArgsTy) );

            llvm::CallInst* CI = b_->CreateCall(TensorMapFn->getFunctionType(), TensorMapFn, llvm::ArrayRef<llvm::Value*>(MapArgs), "llvm_general_map");

            return CI;

        }


    }  // namespace llvm_ir
}  // namespace xla

