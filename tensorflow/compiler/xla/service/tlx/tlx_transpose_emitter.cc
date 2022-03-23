

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/TensorType.h"


#include "tensorflow/compiler/xla/service/tlx/tlx_transpose_emitter.h"
#include "tensorflow/core/platform/logging.h"




namespace xla {
    namespace cpu {




        llvm::CallInst* CreateTransposeCall(llvm::Value* Input, const Shape& InputShape, 
                llvm::Type* ElemTy, llvm::IRBuilder<>* b_){




            llvm::Module* M= b_->GetInsertBlock()->getParent()->getParent();

            llvm::LLVMContext& Ctx = M->getContext();

            int64_t NumElems = GetNumElements(InputShape);


            llvm::VectorType*  VecTy = llvm::FixedVectorType::get(ElemTy, NumElems);


            std::vector<llvm::Type*> TransposeArgsTy = {VecTy};

            llvm::Function* TransposeFn = llvm::Intrinsic::getDeclaration(M, llvm::Intrinsic::tensor_transpose, llvm::ArrayRef<llvm::Type*>(TransposeArgsTy) );


            std::vector<llvm::Value*> TransposeArgs = {Input};

            llvm::CallInst* CI = b_->CreateCall(TransposeFn->getFunctionType(), TransposeFn, llvm::ArrayRef<llvm::Value*>(TransposeArgs), "llvm_transpose");

            return CI;


        }


    }  // namespace llvm_ir
}  // namespace xla

