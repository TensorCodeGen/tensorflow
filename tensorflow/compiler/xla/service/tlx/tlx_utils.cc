#include "tensorflow/compiler/xla/service/tlx/tlx_utils.h"

#include <algorithm>
#include <vector>

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/TensorType.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/TypeSize.h"
#include "tensorflow/compiler/xla/shape.h"

#ifndef TLX_UTILS_CC
#define TLX_UTILS_CC

#define MANTISSA 24
#define EXPONENT 8

// namespace llvm {
//     class TensorType;
// };

namespace xla {
    namespace cpu {

        int64_t GetNumElements(const Shape& TensorShape) {
            int64_t num_elem = 1;

            for (unsigned i : TensorShape.dimensions()) {
                num_elem *= i;
            }

            return num_elem;
        }

        llvm::Value* GetShapeVector(const Shape& TensorShape, llvm::LLVMContext* C) {
            LOG(INFO) << "Invoked GetShapeVector ..."
                << "\n";
            llvm::Type* I32Ty = llvm::Type::getInt32Ty(*C);

            std::vector<llvm::Constant*> ConstShapes;
            for (unsigned i : TensorShape.dimensions()) {
                LOG(INFO) << i << " ";
                ConstShapes.push_back(llvm::ConstantInt::get(I32Ty, i));
            }

            if (ConstShapes.size() <= 1) {
                LOG(INFO) << "Represent < 2D tensor with degenerated 2D Shape ...";
                ConstShapes.insert(ConstShapes.begin(), llvm::ConstantInt::get(I32Ty, 1));
            }

            llvm::Constant* ShapeVector =
                llvm::ConstantVector::get(llvm::ArrayRef<llvm::Constant*>(ConstShapes));

            return ShapeVector;
        }

        llvm::Value* GetReverseShapeVector(const Shape& TensorShape,
                llvm::LLVMContext* C) {
            LOG(INFO) << "Invoked GetReverseShapeVector ..."
                << "\n";
            llvm::Type* I32Ty = llvm::Type::getInt32Ty(*C);

            std::vector<llvm::Constant*> ConstShapes;
            for (unsigned i : TensorShape.dimensions()) {
                LOG(INFO) << i << " ";
                ConstShapes.push_back(llvm::ConstantInt::get(I32Ty, i));
            }

            if (ConstShapes.size() <= 1) {
                LOG(INFO) << "Represent 1D tensor with degenerated 2D Shape ...";
                ConstShapes.insert(ConstShapes.begin(), llvm::ConstantInt::get(I32Ty, 1));
            }

            std::reverse(ConstShapes.begin(), ConstShapes.end());

            llvm::Constant* ShapeVector =
                llvm::ConstantVector::get(llvm::ArrayRef<llvm::Constant*>(ConstShapes));

            return ShapeVector;
        }

        llvm::Value* GetLayoutVector(const Shape& TensorShape, llvm::LLVMContext* C) {
            if (!TensorShape.has_layout()) {
                return nullptr;
            }

            const Layout& L = TensorShape.layout();

            if (L.format() != DENSE) {
                return nullptr;
            }

            std::vector<llvm::Constant*> LayoutArray;

            llvm::Type* I32Ty = llvm::Type::getInt32Ty(*C);
            llvm::Constant* Zero = llvm::ConstantInt::get(I32Ty, 0);
            llvm::Constant* One = llvm::ConstantInt::get(I32Ty, 1);

            for (auto i : L.minor_to_major()) {
                LayoutArray.insert(LayoutArray.begin(), llvm::ConstantInt::get(I32Ty, i));
            }

            if (LayoutArray.size() <= 1) {
                LOG(INFO) << "Represent <= 1D tensor with degenerated 2D Layout ...";
                LayoutArray.insert(LayoutArray.begin(), llvm::ConstantInt::get(I32Ty, 1));
            }

            llvm::Constant* LayoutVector =
                llvm::ConstantVector::get(llvm::ArrayRef<llvm::Constant*>(LayoutArray));

            return LayoutVector;
        }

        llvm::Value* Get0PaddingVector(const Shape& TensorShape, llvm::LLVMContext* C) {
            unsigned NumDim = TensorShape.dimensions_size();

            if (NumDim <= 1) NumDim++;

            llvm::Type* I32Ty = llvm::Type::getInt32Ty(*C);

            llvm::ElementCount EC = llvm::ElementCount::getFixed(NumDim);
            llvm::Constant* PaddingVector =
                llvm::ConstantVector::getSplat(EC, llvm::ConstantInt::get(I32Ty, 0));

            return PaddingVector;
        }

        llvm::Value* LoadPtrToVectorTy(llvm::Value* ArrayPtr, llvm::Type* ScalarTy,
                int64_t NumElems, llvm::IRBuilder<>* b_) {
            LOG(INFO) << "LoadPtrToVectorTy";

            llvm::VectorType* VecTy = llvm::FixedVectorType::get(ScalarTy, NumElems);
            assert(ArrayPtr && "Must provide buffer to load vector from");
            unsigned AS =
                llvm::dyn_cast<llvm::PointerType>(ArrayPtr->getType())->getAddressSpace();
            llvm::PointerType* VecPtrTy = llvm::PointerType::get(VecTy, AS);

            assert(VecPtrTy && "VecPtrTy is NULL");


            llvm::Value* PtrCast = b_->CreatePointerCast(ArrayPtr, VecPtrTy, "vec_cast");
            llvm::Value* VecLoad = b_->CreateLoad(VecTy, PtrCast, "vec.load");

            return VecLoad;
        }

        llvm::StoreInst* StoreVectorTyToPtr(llvm::Value* Vector, llvm::Value* ArrayPtr,
                llvm::Type* ScalarTy, int64_t NumElems,
                llvm::IRBuilder<>* b_) {
            LOG(INFO) << "Creating Store vector inst"
                << "\n";

            llvm::VectorType* VecTy = llvm::FixedVectorType::get(ScalarTy, NumElems);
            unsigned AS =
                llvm::dyn_cast<llvm::PointerType>(ArrayPtr->getType())->getAddressSpace();

            LOG(INFO) << "Obtained address space:\t" << AS << "\n";

            llvm::PointerType* VecPtrTy = llvm::PointerType::get(VecTy, AS);

            llvm::Value* PtrCast =
                b_->CreatePointerCast(ArrayPtr, VecPtrTy, "vec.store.cast");

            LOG(INFO) << "Obtained Ptrcast, now storing back"
                << "\n";
            llvm::StoreInst* VecStore = b_->CreateStore(Vector, PtrCast);

            return VecStore;
        }

        llvm::CallInst* CreateMatMulCall(llvm::Value* lhs, llvm::Value* rhs,
                llvm::Type* TargetType,
                llvm::IRBuilder<>* b_) {
            llvm::Module* M = b_->GetInsertBlock()->getParent()->getParent();

            std::vector<llvm::Type*> MatMulArgsTy = {TargetType};

            llvm::Function* MatMulFn = llvm::Intrinsic::getDeclaration(
                    M, llvm::Intrinsic::tensor_matmul,
                    llvm::ArrayRef<llvm::Type*>(MatMulArgsTy));

            std::vector<llvm::Value*> MatMulArgs = {lhs, rhs};

            llvm::CallInst* CI =
                b_->CreateCall(MatMulFn->getFunctionType(), MatMulFn,
                        llvm::ArrayRef<llvm::Value*>(MatMulArgs), "llvm_matmul");

            return CI;
        }

        llvm::CallInst* CreateConvCall(llvm::Value* lhs, llvm::Value* rhs,
                llvm::Type* TargetType, llvm::Value* strides,
                llvm::Value* lhs_dilations,
                llvm::Value* rhs_dilations,
                llvm::IRBuilder<>* b_) {
            llvm::Module* M = b_->GetInsertBlock()->getParent()->getParent();

            std::vector<llvm::Type*> ConvArgsTy = {TargetType, strides->getType(),
                lhs_dilations->getType(),
                rhs_dilations->getType()};

            llvm::Function* ConvFn =
                llvm::Intrinsic::getDeclaration(M, llvm::Intrinsic::tensor_convolution,
                        llvm::ArrayRef<llvm::Type*>(ConvArgsTy));

            std::vector<llvm::Value*> ConvArgs = {lhs, rhs, strides, lhs_dilations,
                rhs_dilations};

            llvm::CallInst* CI =
                b_->CreateCall(ConvFn->getFunctionType(), ConvFn,
                        llvm::ArrayRef<llvm::Value*>(ConvArgs), "llvm_conv");

            return CI;
        }

        llvm::CallInst* CreateTensorStoreCall(llvm::Value* Token, llvm::Value* Ptr,
                llvm::Value* Stride,
                llvm::IRBuilder<>* b_) {
            llvm::Module* M = b_->GetInsertBlock()->getParent()->getParent();

            std::vector<llvm::Type*> TensorStoreTy = {Ptr->getType(), Stride->getType()};

            llvm::Function* TensorStoreFn = llvm::Intrinsic::getDeclaration(
                    M, llvm::Intrinsic::tensor_store,
                    llvm::ArrayRef<llvm::Type*>(TensorStoreTy));

            std::vector<llvm::Value*> TensorStoreArgs = {Ptr, Stride, Token};

            llvm::CallInst* CI =
                b_->CreateCall(TensorStoreFn->getFunctionType(), TensorStoreFn,
                        llvm::ArrayRef<llvm::Value*>(TensorStoreArgs), "");

            return CI;
        }

        llvm::CallInst* CreateTensorLoadCall(llvm::Value* Ptr, llvm::Value* Shape,
                llvm::Value* Layout, llvm::Value* Padding,
                llvm::Value* Stride,
                llvm::IRBuilder<>* b_) {
            llvm::Module* M = b_->GetInsertBlock()->getParent()->getParent();

            std::vector<llvm::Type*> TensorLoadTy = {
                Ptr->getType(), Shape->getType(), Layout->getType(), Padding->getType(),
                Stride->getType()};

            llvm::Function* TensorLoadFn = llvm::Intrinsic::getDeclaration(
                    M, llvm::Intrinsic::tensor_load,
                    llvm::ArrayRef<llvm::Type*>(TensorLoadTy));

            std::vector<llvm::Value*> TensorLoadArgs = {Ptr, Shape, Layout, Padding,
                Stride};

            llvm::CallInst* CI = b_->CreateCall(
                    TensorLoadFn->getFunctionType(), TensorLoadFn,
                    llvm::ArrayRef<llvm::Value*>(TensorLoadArgs), "llvm_tensor_load");

            return CI;
        }

        llvm::CallInst* CreateTypeInfoCall(llvm::Value* Vector, llvm::Value* Shape,
                llvm::Value* Layout, llvm::Value* Padding,
                llvm::IRBuilder<>* b_) {
            llvm::Module* M = b_->GetInsertBlock()->getParent()->getParent();

            llvm::LLVMContext& Ctx = M->getContext();
            llvm::Type* I32Ty = llvm::Type::getInt32Ty(Ctx);

            llvm::Constant* Mantissa = llvm::ConstantInt::get(I32Ty, MANTISSA);
            llvm::Constant* Exponent = llvm::ConstantInt::get(I32Ty, EXPONENT);

            std::vector<llvm::Type*> TypeInfoArgsTy = {
                Vector->getType(), Shape->getType(), Layout->getType(),
                Padding->getType()};

            llvm::Function* TypeInfoFn = llvm::Intrinsic::getDeclaration(
                    M, llvm::Intrinsic::tensor_typeinfo,
                    llvm::ArrayRef<llvm::Type*>(TypeInfoArgsTy));

            std::vector<llvm::Value*> TypeInfoArgs = {Vector,  Shape,    Layout,
                Padding, Mantissa, Exponent};

            llvm::CallInst* CI = b_->CreateCall(
                    TypeInfoFn->getFunctionType(), TypeInfoFn,
                    llvm::ArrayRef<llvm::Value*>(TypeInfoArgs), "llvm_typeinfo");

            return CI;
        }


        llvm::Value* CastInputToFloat(llvm::Value* Input, llvm::IRBuilder<>* b_){

            llvm::Module* M = b_->GetInsertBlock()->getParent()->getParent();

            llvm::LLVMContext& Ctx = M->getContext();
            switch (Input->getType()->getTypeID()) {
                case llvm::Type::IntegerTyID:
                    return b_->CreateSIToFP(Input, llvm::Type::getFloatTy(Ctx));                
                case llvm::Type::FloatTyID:
                case llvm::Type::DoubleTyID:
                    return Input;
                case llvm::Type::HalfTyID:
                case llvm::Type::BFloatTyID:
                default:
                    assert(false && "Invalid element type.");

            }
            return nullptr;
        }


        llvm::Value* ConvertFloatToType(llvm::Value* Input, llvm::Type* ElemTy , llvm::IRBuilder<>* b_){
            llvm::Module* M = b_->GetInsertBlock()->getParent()->getParent();

            llvm::LLVMContext& Ctx = M->getContext();
            assert(Input->getType()->isFloatingPointTy() && "Conversion input must be floating point type");


            // If the input type is the same
            // as the type we're casting to,
            // simply return the input
            if(Input->getType() == ElemTy){
                return Input;
            }

            switch (ElemTy->getTypeID()) {
                case llvm::Type::IntegerTyID:
                    return b_->CreateFPToSI(Input, ElemTy);                
                default:
                    assert(false && "Invalid element type.");

            }
            return nullptr;

        }

        llvm::Constant* GetConstantValue(llvm::LLVMContext & Ctx, llvm::Type* Ty, int64_t Val){

            switch (Ty->getTypeID()) {
                case llvm::Type::IntegerTyID:
                    return llvm::ConstantInt::get(llvm::Type::getInt32Ty(Ctx), (int)Val);
                case llvm::Type::FloatTyID:
                    return llvm::ConstantFP::get(llvm::Type::getFloatTy(Ctx), (float)Val);
                case llvm::Type::DoubleTyID:
                    return llvm::ConstantFP::get(llvm::Type::getDoubleTy(Ctx), (double)Val);
                case llvm::Type::HalfTyID:
                case llvm::Type::BFloatTyID:
                default:
                    assert(false && "Invalid element type.");

            }
            return nullptr;
        }

    }  // namespace cpu
}  // namespace xla

#endif  // TLX_UTILS_CC
