#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/TensorType.h"


#include "tensorflow/compiler/xla/service/tlx/tlx_gelu_emitter.h"
#include "tensorflow/compiler/xla/service/tlx/tlx_tanh_emitter.h"
#include "tensorflow/compiler/xla/service/tlx/tlx_map_emitter.h"
#include "tensorflow/core/platform/logging.h"


namespace xla {
    namespace cpu {

        // Implementation of the GELU activation according to: https://arxiv.org/abs/1606.08415
        llvm::Function* CreateApproximateGeluElementFunction(llvm::Type* ElemTy, llvm::IRBuilder<>* b_){


            llvm::Module* M= b_->GetInsertBlock()->getParent()->getParent();

            // If function already defined with the given type then return this existing 
            // function
            if(llvm::Function* tlx_gelu =  M->getFunction("tlx_gelu")){
                llvm::Value* Input = &*tlx_gelu->arg_begin();
                if(Input->getType() == ElemTy){
                    return tlx_gelu;
                }
            }

            LOG(INFO) << "[TLX]" << "Creating Approximate GeLU function" << "\n";

            // As we're creating a new function, 
            // the insert point of the IR Builder
            // would be moved hence we save it.
            auto InsertPoint = b_ -> saveIP();


            llvm::LLVMContext& Ctx = M->getContext();

            llvm::FunctionCallee gelu_fun =  M->getOrInsertFunction("tlx_gelu", /* Return Type */ ElemTy, /* Input Type */   ElemTy);



            // Set defult calling convention. Should we set this to be Fast instead?
            llvm::Function* gelu = llvm::cast<llvm::Function>(gelu_fun.getCallee());
            gelu->setCallingConv(llvm::CallingConv::C);


            llvm::Value* Input = &*gelu->arg_begin();
            Input->setName("input");

            llvm::BasicBlock* EntryBB = llvm::BasicBlock::Create(Ctx, "entry", gelu);

            b_->SetInsertPoint(EntryBB);

            llvm::Value* FPInput = CastInputToFloat(Input, b_);

            // Constant values used in GeLU expression
            llvm::Constant *X3Coeff = GetConstantValueFloat(Ctx, FPInput->getType(), (float) 0.044715);
            llvm::Constant *Point5 = GetConstantValueFloat(Ctx, FPInput->getType(), (float) 0.5);
            // sqrt(2/pi)
            llvm::Constant *SqrtCoeff= GetConstantValueFloat(Ctx, FPInput->getType(), (float) 0.7978845608 );
            llvm::Constant *One = GetConstantValue(Ctx, FPInput->getType(), 1);
            llvm::Constant *Three = GetConstantValue(Ctx, FPInput->getType(), 3);


            // =========== Generate the expression to calculate gelu(x)

            // x^3
            std::vector<llvm::Type*> PowArgsTy = {FPInput->getType()};
            llvm::Function* PowFn = llvm::Intrinsic::getDeclaration(M, llvm::Intrinsic::pow, llvm::ArrayRef<llvm::Type*>(PowArgsTy));
            std::vector<llvm::Value*> PowArgs = {FPInput, Three};
            llvm::CallInst* XPow3 = b_->CreateCall(PowFn->getFunctionType(), PowFn, llvm::ArrayRef<llvm::Value*>(PowArgs),
                    "x_pow_3");

            // x^3 * 0.044715
            llvm::Value* Expr = b_->CreateFMul(X3Coeff, XPow3, "");


            // x + (x^3 * 0.044715)
            Expr = b_->CreateFAdd(FPInput, Expr, "");

            // sqrt(2/pi) * [x + (x^3 * 0.044715)]
            Expr = b_->CreateFMul(SqrtCoeff , Expr, "");

            // tanh((sqrt(2/pi) * [x + (x^3 * 0.044715)]))
            llvm::Function* tlx_tanh = CreateTanhElementFunction(FPInput->getType(), b_);
            

            std::vector<llvm::Value*> TanHArgs = {Expr};
            Expr = b_->CreateCall(tlx_tanh->getFunctionType(), tlx_tanh, llvm::ArrayRef<llvm::Value*>(TanHArgs), "tanh" ) ;


            // 1 +  tanh((sqrt(2/pi) * [x + (x^3 * 0.044715)]))
            Expr = b_->CreateFAdd(One , Expr, "");


            // x * (1 +  tanh((sqrt(2/pi) * [x + (x^3 * 0.044715)])))
            Expr = b_->CreateFMul(FPInput , Expr, "");
            
            // 0.5 * x * (1 +  tanh((sqrt(2/pi) * [x + (x^3 * 0.044715)])))
            Expr = b_->CreateFMul(Point5 , Expr, "gelu");


            llvm::Value* CastBack = ConvertFloatToType(Expr, ElemTy, b_ );

            llvm::Instruction* Return =  b_->CreateRet(CastBack);




            // Return the IRBuilder to the previously
            // saved insert point
            b_ ->restoreIP(InsertPoint);

            return gelu;
        }


        void EmitTLXGelu_Helper(const llvm_ir::IrArray& source_array_, const llvm_ir::IrArray& target_array_, llvm::IRBuilder<>* b_){


            LOG(INFO) << "[TLX]\t" << "Emit TLX Gelu Helper"<<"\n";


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


            llvm::Function* Gelu = CreateApproximateGeluElementFunction(SourceElemType, b_);

            llvm::Value* Gelu_vector = CreateMapCall(source_type_info, Gelu, TargetVecTy, b_);


            LOG(INFO) << "[TLX]\t" << "Create tensor gelu typeinfo call"<<"\n";

            llvm::CallInst* target_type_info = CreateTypeInfoCall(Gelu_vector, tlx_source_shape, tlx_source_layout, tlx_source_padding, b_);


            LOG(INFO) << "[TLX]\t" << "Create store back for result"<<"\n";


            llvm::StoreInst* StoreResult = StoreVectorTyToPtr(Gelu_vector, target_ptr, TargetElemType, num_target_values , b_ );



            LOG(INFO) << "[TLX]\t" << "Completed generation of TLX Gelu "<<"\n";

        }


    }
}
