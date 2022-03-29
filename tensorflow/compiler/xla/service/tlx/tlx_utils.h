#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/TypeSize.h"

#include "tensorflow/compiler/xla/shape.h"

#include "tensorflow/core/platform/logging.h"
#include "llvm/IR/TensorType.h"
#include <vector>




//namespace llvm {
//    class TensorType;
//};

namespace xla {
namespace cpu {

int64_t GetNumElements(const Shape& TensorShape);


llvm::Value* GetShapeVector(const Shape& TensorShape, llvm::LLVMContext* C);


llvm::Value* GetLayoutVector(const Shape& TensorShape, llvm::LLVMContext* C);

llvm::Value* Get0PaddingVector(const Shape& TensorShape, llvm::LLVMContext* C);


llvm::Value* LoadPtrToVectorTy(llvm::Value* ArrayPtr, llvm::Type* ScalarTy,  int64_t NumElems, llvm::IRBuilder<>* b_);


llvm::StoreInst* StoreVectorTyToPtr(llvm::Value* Vector ,llvm::Value* ArrayPtr, llvm::Type* ScalarTy,  int64_t NumElems, llvm::IRBuilder<>* b_);



llvm::CallInst* CreateMatMulCall(llvm::Value* lhs, llvm::Value* rhs, llvm::Type* TargetType , llvm::IRBuilder<>* b_);


llvm::CallInst* CreateTypeInfoCall(llvm::Value* Vector, llvm::Value* Shape, llvm::Value* Layout, llvm::Value* Padding, llvm::IRBuilder<>* b_);


llvm::CallInst* CreateTensorStoreCall(llvm::Value* Token, llvm::Value* Ptr, llvm::Value* Stride  ,llvm::IRBuilder<>* b_);


llvm::CallInst* CreateTensorLoadCall( llvm::Value* Ptr, llvm::Value* Shape, llvm::Value* Layout, llvm::Value* Padding ,llvm::Value* Stride , llvm::IRBuilder<>* b_);

}  // namespace llvm_ir
}
