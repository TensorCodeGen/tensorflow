#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/TypeSize.h"

#include "tensorflow/compiler/xla/shape.h"
#include "llvm/IR/TensorType.h"
#include <vector>




//namespace llvm {
//    class TensorType;
//};

namespace xla {
namespace cpu {




llvm::Value* GetShapeVector(const Shape& TensorShape, llvm::LLVMContext* C){
    llvm::Type* I32Ty = llvm::Type::getInt32Ty(*C);

    std::vector<llvm::Constant*> ConstShapes;
    for(unsigned i : TensorShape.dimensions()){
        ConstShapes.push_back(
                llvm::ConstantInt::get(I32Ty, i)
                );
    }

    llvm::Constant* ShapeVector = llvm::ConstantVector::get(llvm::ArrayRef<llvm::Constant*>(ConstShapes));

    return ShapeVector;

}


llvm::Value* GetLayoutVector(const Shape& TensorShape, llvm::LLVMContext* C){
    if (!TensorShape.has_layout()){
        return nullptr;
    }

    const Layout& L = TensorShape.layout();

    if(L.format() != DENSE){
        return nullptr;
    }

    std::vector<llvm::Constant*> LayoutArray;



    llvm::Type* I32Ty = llvm::Type::getInt32Ty(*C);
    llvm::Constant* Zero =  llvm::ConstantInt::get(I32Ty, 0);
    llvm::Constant* One =  llvm::ConstantInt::get(I32Ty, 1);

    /*
    if(L.minor_to_major()){
        LayoutArray.push_back(One);
        LayoutArray.push_back(Zero);
    } else if (Layout.major_to_minor()){
        LayoutArray.push_back(Zero);
        LayoutArray.push_back(One);
    }*/

    for(auto i : L.minor_to_major()){
        LayoutArray.push_back(
                llvm::ConstantInt::get(I32Ty, i)
                );
    }



    llvm::Constant* LayoutVector = llvm::ConstantVector::get(llvm::ArrayRef<llvm::Constant*>(LayoutArray));

    return LayoutVector;

}

llvm::Value* Get0PaddingVector(const Shape& TensorShape, llvm::LLVMContext* C){
    unsigned NumDim = TensorShape.dimensions_size();


    llvm::Type* I32Ty = llvm::Type::getInt32Ty(*C);

    llvm::ElementCount EC = llvm::ElementCount::getFixed(NumDim);
    llvm::Constant* PaddingVector = llvm::ConstantVector::getSplat(EC, 
            llvm::ConstantInt::get(I32Ty, 0));

    return PaddingVector;

}


llvm::CallInst* CreateTypeInfoCall(llvm::Value* Vector, llvm::Value* Shape, llvm::Value* Layout, llvm::Value* Padding, llvm::IRBuilder<>* b_){

    llvm::Module* M= b_->GetInsertBlock()->getParent()->getParent();

    std::vector<llvm::Type*> TypeInfoArgsTy = {Vector->getType(), Shape->getType(), Layout->getType(), Padding->getType()};

    llvm::Function* TypeInfoFn = llvm::Intrinsic::getDeclaration(M, llvm::Intrinsic::tensor_typeinfo, llvm::ArrayRef<llvm::Type*>(TypeInfoArgsTy) );


    std::vector<llvm::Value*> TypeInfoArgs = {Vector, Shape, Layout, Padding};

    llvm::CallInst* CI = b_->CreateCall(TypeInfoFn->getFunctionType(), TypeInfoFn, llvm::ArrayRef<llvm::Value*>(TypeInfoArgs), "llvm_typeinfo");

    return CI;

}

}  // namespace llvm_ir
}
