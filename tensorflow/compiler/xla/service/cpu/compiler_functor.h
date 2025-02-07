/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_COMPILER_FUNCTOR_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_COMPILER_FUNCTOR_H_

#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "tensorflow/compiler/xla/service/llvm_compiler.h"
#include "tensorflow/core/platform/logging.h"



namespace xla {
namespace cpu {

// Functor class for compiling an LLVM module down to an object file. For use by
// Orc JIT compile layer.
class CompilerFunctor : public llvm::orc::IRCompileLayer::IRCompiler {
 public:
  explicit CompilerFunctor(
      llvm::TargetMachine* target_machine, int opt_level,
      bool optimize_for_size, bool disable_expensive_passes,
      llvm::FastMathFlags fast_math_flags,
      LLVMCompiler::ModuleHook pre_optimization_hook = nullptr,
      LLVMCompiler::ModuleHook post_optimization_hook = nullptr,
      std::function<void(const llvm::object::ObjectFile&)> post_codegen_hook =
          nullptr)
      : IRCompiler(llvm::orc::IRSymbolMapper::ManglingOptions()),
        target_machine_(target_machine),
        opt_level_(opt_level),
        optimize_for_size_(optimize_for_size),
        disable_expensive_passes_(disable_expensive_passes),
        fast_math_flags_(fast_math_flags),
        pre_optimization_hook_(std::move(pre_optimization_hook)),
        post_optimization_hook_(std::move(post_optimization_hook)),
        post_codegen_hook_(std::move(post_codegen_hook)) {}

  // Compile a Module to an ObjectFile.
  llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>> operator()(
      llvm::Module& module) override;

 private:
  // Populates the given pass manager with TargetLibraryInfo and
  // TargetTransformInfo passes.
  void AddTargetInfoPasses(llvm::legacy::PassManagerBase* passes) const;



  // Populates the given pass managers based on the optimization level.
  void AddOptimizationPasses(llvm::legacy::PassManagerBase* module_passes,
                             llvm::legacy::FunctionPassManager* function_passes,
                             unsigned opt_level, unsigned size_level) const;


  void AddTLXPasses(llvm::legacy::FunctionPassManager* passes) const;

  llvm::TargetMachine* target_machine_;
  const unsigned opt_level_;
  const bool optimize_for_size_;
  const bool disable_expensive_passes_;
  const llvm::FastMathFlags fast_math_flags_;
  LLVMCompiler::ModuleHook pre_optimization_hook_;
  LLVMCompiler::ModuleHook post_optimization_hook_;
  std::function<void(const llvm::object::ObjectFile&)> post_codegen_hook_;
};

}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_COMPILER_FUNCTOR_H_
