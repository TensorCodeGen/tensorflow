#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Type.h"

#include "llvm/IR/TensorType.h"

#include "tensorflow/compiler/xla/shape.h"

#include <vector>

llvm::Value* GetShapeVector(const &Shape TensorShape, llvm::LLVMContext* C){
    llvm::Type* I32Ty = llvm::Type::getInt32Ty(*C);

    std::vector<llvm::Constant*> ConstShapes;
    for(unsigned i : TensorShape.dimensions()){
        ConstShapes.push_back(
                llvm::ConstantInt::get(I32Ty, i);
                );
    }

    llvm::Constant* ShapeVector = llvm::ConstantVector::get(ArrayRef(ConstShapes));

    return ShapeVector;

}


llvm::Value* GetLayoutVector(const &Shape TensorShape, llvm::LLVMContext* C){
    if (!TensorShape.has_layout()){
        return nullptr;
    }

    const &Layout = TensorShape.layout();


}

