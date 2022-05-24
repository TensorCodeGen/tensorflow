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


    }
}
