/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

// This file implements logic for lowering HLO dialect to LHLO dialect.

#include <functional>
#include <memory>
#include <utility>

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir-hlo/Dialect/mhlo/transforms/type_conversion.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace mhlo {
namespace {

// Wrap `value` in a ToMemrefOp. If `value` is a ToTennsorOp, `value` was
// already bufferized. In that case, take the memref operand from that op.
// TODO(springerm): This function will disappear once the RewritePatterns in
// this function become BufferizableOpInterface implementation.
static Value wrapInToMemrefOp(RewriterBase& rewriter, Value value) {
  bufferization::BufferizationOptions options;
  options.fullyDynamicLayoutMaps = false;
  return bufferization::lookupBuffer(rewriter, value, options);
}

// TODO(springerm): Turn these rewrite patterns into BufferizableOpInterface
// implementations.
template <typename T>
class SignlessOpConversion : public OpRewritePattern<T> {
 public:
  SignlessOpConversion(RemoveSignTypeConverter* remove_sign_converter,
                       MLIRContext* ctx)
      : OpRewritePattern<T>(ctx),
        remove_sign_converter_(remove_sign_converter) {}

  Value convertToSignless(RewriterBase& rewriter, Location loc,
                          Value value) const {
    Type original = value.getType();
    Type converted = remove_sign_converter_->convertType(original);
    if (converted == original) {
      return value;
    } else {
      return rewriter.create<UnrealizedConversionCastOp>(loc, converted, value)
          ->getResult(0);
    }
  }

  LogicalResult matchAndRewrite(T op, PatternRewriter& rewriter) const final {
    auto loc = op.getLoc();
    // Sign-convert result type.
    Type op_result_type = remove_sign_converter_->convertType(op.getType());
    // Perform actual rewrite.
    Value result = signlessRewrite(op, op_result_type, rewriter);
    if (!result) return failure();

    // If the element type of the original op and the returned value differ,
    // do a conversion cast to fix it up.
    auto expected_element_type =
        op.getType().template cast<ShapedType>().getElementType();
    auto result_type = result.getType().cast<BaseMemRefType>();
    auto actual_element_type = result_type.getElementType();
    if (expected_element_type != actual_element_type) {
      assert(remove_sign_converter_->convertType(expected_element_type) ==
             actual_element_type);
      Type new_type;
      if (auto ranked = result_type.dyn_cast<MemRefType>()) {
        new_type = MemRefType::get(ranked.getShape(), expected_element_type,
                                   ranked.getLayout(), ranked.getMemorySpace());
      } else {
        new_type = UnrankedMemRefType::get(expected_element_type,
                                           result_type.getMemorySpace());
      }
      result =
          rewriter.create<UnrealizedConversionCastOp>(loc, new_type, result)
              .getResult(0);
    }
    bufferization::replaceOpWithBufferizedValues(rewriter, op, result);
    return success();
  }

 protected:
  virtual Value signlessRewrite(T op, Type result_type,
                                RewriterBase& rewriter) const = 0;

 private:
  RemoveSignTypeConverter* remove_sign_converter_;
};

template <typename T>
using BaseOpConversion = SignlessOpConversion<T>;

class HloToMemrefReshapeUnrankedConverter
    : public BaseOpConversion<mhlo::ReshapeOp> {
 public:
  using BaseOpConversion<mhlo::ReshapeOp>::BaseOpConversion;

  Value signlessRewrite(mhlo::ReshapeOp op, Type op_result_type,
                        RewriterBase& rewriter) const final {
    auto unranked_operand_type =
        op.operand().getType().dyn_cast<UnrankedTensorType>();
    if (unranked_operand_type == nullptr) return {};
    auto loc = op->getLoc();
    auto result_type = op_result_type.cast<RankedTensorType>();

    Value operand_buffer = wrapInToMemrefOp(rewriter, op.operand());
    auto cast = rewriter.create<memref::CastOp>(
        loc,
        MemRefType::get(result_type.getShape(), result_type.getElementType()),
        convertToSignless(rewriter, loc, operand_buffer));

    return cast;
  }
};

class HloToMemrefDynamicReshapeConverter
    : public BaseOpConversion<mhlo::DynamicReshapeOp> {
 public:
  using BaseOpConversion<mhlo::DynamicReshapeOp>::BaseOpConversion;

  Value signlessRewrite(mhlo::DynamicReshapeOp op, Type op_result_type,
                        RewriterBase& rewriter) const final {
    ShapedType result_type;
    if (auto ranked_type = op_result_type.dyn_cast<RankedTensorType>()) {
      result_type =
          MemRefType::get(ranked_type.getShape(), ranked_type.getElementType());
    } else if (auto unranked_type =
                   op_result_type.dyn_cast<UnrankedTensorType>()) {
      result_type = UnrankedMemRefType::get(unranked_type.getElementType(), 0);
    } else {
      return {};
    }

    Value operand_buffer = wrapInToMemrefOp(rewriter, op.operand());
    Value shape_buffer = wrapInToMemrefOp(rewriter, op.output_shape());
    auto reshape = rewriter.create<memref::ReshapeOp>(
        op.getLoc(), result_type,
        convertToSignless(rewriter, op.getLoc(), operand_buffer),
        convertToSignless(rewriter, op.getLoc(), shape_buffer));
    return reshape;
  }
};

// TODO(b/175670649) Fix this to no longer access original tensor operands.
class HloToMemrefDynamicBroadcastInDimOpConverter
    : public BaseOpConversion<mhlo::DynamicBroadcastInDimOp> {
 public:
  HloToMemrefDynamicBroadcastInDimOpConverter(
      RemoveSignTypeConverter* sign_converter, MLIRContext* ctx,
      std::function<bool(Operation*)> enforce_identity_maps)
      : BaseOpConversion<mhlo::DynamicBroadcastInDimOp>(sign_converter, ctx),
        enforce_identity_maps_(std::move(enforce_identity_maps)) {}

  Value signlessRewrite(mhlo::DynamicBroadcastInDimOp op, Type op_result_type,
                        RewriterBase& rewriter) const final {
    auto result_type = op_result_type.dyn_cast<RankedTensorType>();
    if (!result_type) return {};

    Value operand_buffer = wrapInToMemrefOp(rewriter, op.operand());
    Value result = InsertDynamicMemrefCastOp(
        op, convertToSignless(rewriter, op.getLoc(), operand_buffer),
        &rewriter);

    if (enforce_identity_maps_(op)) {
      result = CreateCopy(op, result, &rewriter);
    }

    return result;
  }

 private:
  // Inserts dynamic memref to change the layout of the memref to put 0-stride
  // and size of the target dimension if size-1 dimension expansion is
  // necessary.
  memref::ReinterpretCastOp InsertDynamicMemrefCastOp(
      mhlo::DynamicBroadcastInDimOp op, Value operand, OpBuilder* b) const {
    auto loc = op.getLoc();
    auto operand_type = operand.getType().cast<MemRefType>();
    auto operand_shape = operand_type.getShape();
    auto operand_rank = operand_type.getRank();

    auto result_type = op.getType().cast<RankedTensorType>();
    auto result_rank = result_type.getRank();

    Value zero = b->create<arith::ConstantIndexOp>(loc, 0);
    Value one = b->create<arith::ConstantIndexOp>(loc, 1);

    // Compute a reversed scan product. Compute the stride for the dimensions so
    // far, working from minor to major dimensions. Additionally, save the
    // operand shape Values to use in the next loop.
    SmallVector<Value, 2> operand_strides(operand_rank, one);
    SmallVector<Value, 2> operand_sizes(operand_rank, one);
    Value stride_so_far = one;
    for (int i = operand_rank - 1; i >= 0; --i) {
      Value operand_dim_size =
          ShapedType::isDynamic(operand_shape[i])
              ? b->create<memref::DimOp>(loc, operand, i).getResult()
              : b->create<arith::ConstantIndexOp>(loc, operand_shape[i])
                    .getResult();
      operand_sizes[i] = operand_dim_size;

      operand_strides[i] = stride_so_far;
      if (i > 0) {
        stride_so_far =
            b->create<arith::MulIOp>(loc, stride_so_far, operand_dim_size);
      }
    }

    SmallVector<OpFoldResult, 2> sizes, strides;
    sizes.reserve(result_rank);
    strides.reserve(result_rank);

    DenseMap<int, int> output_to_input_dim;
    for (const auto& dim : llvm::enumerate(op.broadcast_dimensions())) {
      output_to_input_dim[dim.value().getSExtValue()] = dim.index();
    }
    for (int i = 0; i < result_rank; ++i) {
      Value i_val = b->create<arith::ConstantIndexOp>(loc, i);
      Value result_dim_size =
          b->create<tensor::ExtractOp>(loc, op.output_dimensions(), i_val);
      if (!result_dim_size.getType().isIndex()) {
        result_dim_size = b->create<arith::IndexCastOp>(loc, b->getIndexType(),
                                                        result_dim_size);
      }
      if (result_type.isDynamicDim(i)) {
        sizes.push_back(result_dim_size);
      } else {
        sizes.push_back(b->getIndexAttr(result_type.getDimSize(i)));
      }

      auto it = output_to_input_dim.find(i);
      // If the rank of the output is greater than the rank of the input, i.e.
      // there was no output dimension in the inverse broadcast_dimensions map
      // we also set stride to 0 to emulate padding of the shape with 1s and the
      // corresponding expansion.
      if (it == output_to_input_dim.end()) {
        strides.push_back(zero);
        continue;
      }

      // There can be two cases:
      // 1) Operand dim == result dim => expansion is not needed
      //    => stride flattened buffer stride
      // 2) Operand dim < result dim => expansion is needed => stride := 0.
      int dim = it->second;
      Value is_expansion = b->create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::slt, operand_sizes[dim], result_dim_size);
      Value select = b->create<mlir::arith::SelectOp>(loc, is_expansion, zero,
                                                      operand_strides[dim]);
      strides.push_back(select);
    }

    // Type-erased memref type with static rank and dynamic strides.
    SmallVector<int64_t, 2> dynamic_layout(result_rank,
                                           ShapedType::kDynamicStrideOrOffset);
    auto type_erased_memref_type = MemRefType::get(
        result_type.getShape(), operand_type.getElementType(),
        makeStridedLinearLayoutMap(dynamic_layout,
                                   /*offset=*/0, b->getContext()));

    auto transformed_operand = b->create<memref::ReinterpretCastOp>(
        loc, type_erased_memref_type, operand,
        /*offset=*/b->getI64IntegerAttr(0), sizes, strides);
    return transformed_operand;
  }

  Value CreateCopy(mhlo::DynamicBroadcastInDimOp op, Value broadcasted,
                   OpBuilder* b) const {
    MemRefType result_type = broadcasted.getType().cast<MemRefType>();
    auto loc = op.getLoc();
    SmallVector<Value, 4> dynamic_operands;
    for (int i = 0; i < result_type.getRank(); ++i) {
      if (!result_type.isDynamicDim(i)) continue;
      auto index = b->createOrFold<arith::ConstantIndexOp>(loc, i);
      Value size =
          b->create<tensor::ExtractOp>(loc, op.output_dimensions(), index);
      if (!size.getType().isIndex()) {
        size = b->create<arith::IndexCastOp>(loc, b->getIndexType(), size);
      }
      dynamic_operands.push_back(size);
    }
    auto identity_map_memref =
        MemRefType::get(result_type.getShape(), result_type.getElementType());
    auto copy = b->create<memref::AllocOp>(op.getLoc(), identity_map_memref,
                                           dynamic_operands);
    b->create<memref::CopyOp>(loc, broadcasted, copy);

    return copy;
  }

  std::function<bool(Operation*)> enforce_identity_maps_;
};

struct HloLegalizeToMemrefPass
    : public HloLegalizeToMemrefPassBase<HloLegalizeToMemrefPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<bufferization::BufferizationDialect, memref::MemRefDialect,
                    tensor::TensorDialect>();
  }

 public:
  void runOnOperation() override {
    auto& context = getContext();
    RewritePatternSet patterns(&context);
    RemoveSignTypeConverter sign_converter;
    populateHLOToMemrefConversionPattern(&sign_converter, &patterns);

    auto module = getOperation();
    OpBuilder b(module);
    b.create<memref::GlobalOp>(
        module.getLoc(), b.getStringAttr("rng_state"),
        b.getStringAttr("private"),
        MemRefType::get({}, b.getIntegerType(128, false)),
        b.getIntegerAttr(b.getIntegerType(128, false), 0x7012395ull), false,
        /*alignment=*/IntegerAttr());
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
      signalPassFailure();
  }
};

class HloToMemrefRngGetAndUpdateStateConverter
    : public OpRewritePattern<mhlo::XlaRngGetAndUpdateStateOp> {
 public:
  using OpRewritePattern<mhlo::XlaRngGetAndUpdateStateOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::XlaRngGetAndUpdateStateOp op,
                                PatternRewriter& rewriter) const final {
    // Get various type related information
    auto loc = op->getLoc();

    const auto global_name = rewriter.getStringAttr("rng_state");
    constexpr auto initial_seed = 0x7012395ull;
    auto seed_type = rewriter.getIntegerType(128);
    auto memref_type = MemRefType::get({}, seed_type);

    auto result_type = op.getType();
    auto word_size = result_type.getElementType().getIntOrFloatBitWidth();
    auto smaller_int_type = rewriter.getIntegerType(word_size);
    auto num_elements = result_type.getNumElements();

    // Get or define the global variable
    auto global_op =
        mlir::SymbolTable::lookupNearestSymbolFrom(op, global_name);
    if (!global_op) {
      auto parent = mlir::SymbolTable::getNearestSymbolTable(op);
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(&parent->getRegions().front().front());

      const auto priv = rewriter.getStringAttr("private");
      auto initial_value = mlir::DenseElementsAttr::get(
          mlir::RankedTensorType::get({}, seed_type),
          rewriter.getIntegerAttr(seed_type, initial_seed));
      global_op =
          rewriter.create<memref::GlobalOp>(loc, global_name, priv, memref_type,
                                            initial_value, /*constant=*/false,
                                            /*alignment=*/IntegerAttr());
    }
    assert(isa<memref::GlobalOp>(global_op) &&
           "rng_state was defined somewhere else, not as a global op");

    // Get and update
    Value rng_state =
        rewriter.create<memref::GetGlobalOp>(loc, memref_type, global_name);
    Value old_val = rewriter.create<memref::LoadOp>(loc, rng_state);
    Value delta = rewriter.create<arith::ConstantOp>(
        loc,
        rewriter.getIntegerAttr(seed_type, static_cast<int64_t>(op.delta())));
    Value new_val = rewriter.create<arith::AddIOp>(loc, old_val, delta);
    (void)rewriter.create<memref::StoreOp>(loc, new_val, rng_state);

    // Create the proper return type by packing the old seed into a tensor
    SmallVector<Value> pieces;
    for (int i = (num_elements - 1) * word_size; i >= 0; i -= word_size) {
      Value shift_distance = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIntegerAttr(seed_type, i));
      pieces.push_back(rewriter.create<arith::TruncIOp>(
          loc, smaller_int_type,
          rewriter.create<arith::ShRUIOp>(loc, old_val, shift_distance)));
    }

    // Obtain a tensor with the correct shape and bit widths but the incorrect
    // integer signedness, then cast the tensor to the correct signedness to
    // ensure that unrealized casts will successfully lower later.
    Value result_tensor = rewriter.create<tensor::FromElementsOp>(
        loc,
        mlir::RankedTensorType::get(result_type.getShape(), smaller_int_type),
        pieces);
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(op, result_type,
                                                            result_tensor);
    return success();
  }
};

}  // namespace

void populateHLOToMemrefConversionPattern(
    RemoveSignTypeConverter* sign_converter, RewritePatternSet* patterns,
    const std::function<bool(Operation*)>& enforce_identity_maps) {
  MLIRContext* context = patterns->getContext();
  patterns->add<HloToMemrefDynamicBroadcastInDimOpConverter>(
      sign_converter, context, enforce_identity_maps);
  patterns->add<HloToMemrefRngGetAndUpdateStateConverter>(context);
  patterns->add<HloToMemrefDynamicReshapeConverter,
                HloToMemrefReshapeUnrankedConverter>(sign_converter, context);
}

std::unique_ptr<OperationPass<ModuleOp>> createLegalizeToMemrefPass() {
  return std::make_unique<HloLegalizeToMemrefPass>();
}

}  // namespace mhlo
}  // namespace mlir
