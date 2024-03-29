
#include "codegen_wasm.h"

#include <vector>
#include <memory>

#include <support/file.h>
#include <wasm-s-parser.h>
#include <pass.h>

namespace tvm {
namespace codegen {

CodeGenWasm::CodeGenWasm() = default;
CodeGenWasm::~CodeGenWasm() = default;

void CodeGenWasm::Init() {
  seq_stmt_depth_ = -1;
  module_.reset(new wasm::Module());
  
  DLOG(INFO) << "Load TVM wasm module";
  DLOG(INFO) << "Step 1: Load TVM wasm module file";
  const std::string tvm_wasm_module_filepath = "apps/wasm/tvm_wasm_module.wat";
  auto input(wasm::read_file<std::string>(tvm_wasm_module_filepath, wasm::Flags::Text));
  DLOG(INFO) << "Step 2: Parse wasm expressions";
  wasm::SExpressionParser parser(const_cast<char*>(input.c_str()));
  wasm::Element& root = *parser.root;
  DLOG(INFO) << "Step 3: Build wasm module";
  wasm::SExpressionWasmBuilder builder(*module_, *root[0], wasm::IRProfile::Normal);
  DLOG(INFO) << "TVM wasm module loaded";
  
  builder_.reset(new wasm::Builder(*module_));
  analyzer_.reset(new arith::Analyzer());
  //AddWasmExportedMemory();
}

void CodeGenWasm::AddFunction(const PrimFunc& f) {
  ICHECK_EQ(f->buffer_map.size(), 0U)
      << "Cannot codegen function with buffer_map, please lower them first";

  DLOG(INFO) << f;

  for (size_t i =0; i < f->params.size(); ++ i) {
    const Var& param = f->params[i];
    wasm::NameType name_param(wasm::Name::fromInt(i), GetWasmType(param));
    wasm_params_.push_back(name_param);
    var_map_[param.get()] = name_param;
  }
  std::vector<wasm::Type> params;
  for (auto p : wasm_params_) {
    params.push_back(p.type);
  }
  std::vector<wasm::Type> results;
  results.push_back(wasm::Type(wasm::Type::BasicType::i32));

  auto inline_sig = wasm::Signature(wasm::Type(params), wasm::Type(results));
  wasm::HeapType type(inline_sig);

  stmt_body_exprs_.push_back(std::vector<wasm::Expression*>());

  wasm::Expression* func_body_expr = this->VisitStmt(f->body);

  ICHECK_EQ(stmt_body_exprs_.size(), 1);
  stmt_body_exprs_.back().push_back(func_body_expr);
  stmt_body_exprs_.back().push_back(builder_->makeReturn(CreateWasmInt32Const(0)));
  std::vector<wasm::Expression*> func_body_exprs = stmt_body_exprs_.back();
  stmt_body_exprs_.pop_back();

  for (auto v : wasm_vars_) {
    params.push_back(v.type);
  }

  auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
  ICHECK(global_symbol.defined())
      << "CodeGenLLVM: Expect PrimFunc to have the global_symbol attribute";
  const std::string function_name(global_symbol.value().c_str());
  function_.reset(module_->getFunctionOrNull(function_name));
  if (function_ == nullptr) {
    DLOG(INFO) << "Make function, name " << function_name;
    function_ = builder_->makeFunction(
        function_name,
        std::move(wasm_params_),
        type,
        std::move(wasm_vars_));

    function_->body = builder_->makeBlock(func_body_exprs);

    module_->addFunction(function_.release());

    auto ex = std::make_unique<wasm::Export>();
    ex->name = wasm::Name(function_name);
    ex->value = wasm::Name(function_name);
    ex->kind = wasm::ExternalKind::Function;
    module_->addExport(ex.release());
  }
}

void CodeGenWasm::AddMainFunction(const std::string& entry_func_name) {
}

std::unique_ptr<wasm::Module> CodeGenWasm::Finish() {
  this->Optimize();
  return std::move(module_);
}

wasm::Expression* CodeGenWasm::VisitExpr_(const VarNode* op) {
  DLOG(INFO) << "Visit var node";
  return GetVarValue(op);
}

wasm::Expression* CodeGenWasm::VisitExpr_(const CastNode* op) {
  LOG(FATAL) << "not implemented";
  return nullptr;
}

wasm::Expression* CodeGenWasm::VisitExpr_(const IntImmNode* op) {
  DLOG(INFO) << "Visit int imm node";
  DLOG(INFO) << "op_value " << op->value;
  DataType dtype = op->dtype;
  if (dtype.bits() == 64) {
    return CreateWasmInt64Const(op->value);
  } else {
    return CreateWasmInt32Const(op->value);
  }
}

wasm::Expression* CodeGenWasm::VisitExpr_(const FloatImmNode* op) {
  DLOG(INFO) << "Visit float imm node";
  DLOG(INFO) << "op_value " << op->value;
  return CreateWasmFloatConst(op->value);
}

wasm::Expression* CodeGenWasm::VisitExpr_(const StringImmNode* op) {
  LOG(FATAL) << "not implemented";
  return nullptr;
}

// QUESTION(fucheng): How to know op->a is a scalar or a vector?
#define DEFINE_CODEGEN_BINARY_OP(Op) \
  wasm::Expression* CodeGenWasm::Create##Op(DataType t, wasm::Expression* a, wasm::Expression* b) { \
    if (t.is_int() || t.is_uint()) { \
      return builder_->makeBinary(wasm::BinaryOp::Op##Int32, a, b); \
    } else { \
      ICHECK(t.is_float()); \
      if (t.is_vector()) { \
        return builder_->makeBinary(wasm::BinaryOp::Op##VecF32x4, a, b); \
      } else { \
        return builder_->makeBinary(wasm::BinaryOp::Op##Float32, a, b); \
      } \
    } \
  } \
  wasm::Expression* CodeGenWasm::VisitExpr_(const Op##Node* op) { \
    DLOG(INFO) << "Visit add/sub/mul node"; \
    return Create##Op(op->dtype, MakeValue(op->a), MakeValue(op->b)); \
  }

DEFINE_CODEGEN_BINARY_OP(Add);
DEFINE_CODEGEN_BINARY_OP(Sub);
DEFINE_CODEGEN_BINARY_OP(Mul);

wasm::Expression* CodeGenWasm::VisitExpr_(const DivNode* op) {
  DLOG(INFO) << "Visit div node";
  wasm::Expression* a = MakeValue(op->a);
  wasm::Expression* b = MakeValue(op->b);
  if (op->dtype.is_int()) {
    return builder_->makeBinary(wasm::BinaryOp::DivSInt32, a, b);
  } else if (op->dtype.is_uint()) {
    return builder_->makeBinary(wasm::BinaryOp::DivUInt32, a, b);
  } else {
    ICHECK(op->dtype.is_float());
    return builder_->makeBinary(wasm::BinaryOp::DivFloat32, a, b);
  }
}

wasm::Expression* CodeGenWasm::VisitExpr_(const ModNode* op) {
  DLOG(INFO) << "Visit mod node";
  wasm::Expression* a = MakeValue(op->a);
  wasm::Expression* b = MakeValue(op->b);
  if (op->dtype.is_int()) {
    return builder_->makeBinary(wasm::BinaryOp::RemSInt32, a, b);
  } else if (op->dtype.is_uint()) {
    return builder_->makeBinary(wasm::BinaryOp::RemUInt32, a, b);
  } else {
    LOG(FATAL) << "not implemented";
    return nullptr;
  }
}

wasm::Expression* CodeGenWasm::VisitExpr_(const MinNode* op) {
  DLOG(INFO) << "Visit min node";
  ICHECK(op->dtype.is_float());
  wasm::Expression* a = MakeValue(op->a);
  wasm::Expression* b = MakeValue(op->b);
  return builder_->makeBinary(wasm::BinaryOp::MinFloat32, a, b);
}

wasm::Expression* CodeGenWasm::VisitExpr_(const MaxNode* op) {
  DLOG(INFO) << "Visit max node";
  ICHECK(op->dtype.is_float());
  wasm::Expression* a = MakeValue(op->a);
  wasm::Expression* b = MakeValue(op->b);
  return builder_->makeBinary(wasm::BinaryOp::MaxFloat32, a, b);
}

wasm::Expression* CodeGenWasm::VisitExpr_(const LTNode* op) {
  DLOG(INFO) << "Visit lt node";
  wasm::Expression* a = MakeValue(op->a);
  wasm::Expression* b = MakeValue(op->b);
  if (op->dtype.is_int()) {
    return builder_->makeBinary(wasm::BinaryOp::LtSInt32, a, b);
  } else if (op->dtype.is_uint()) {
    return builder_->makeBinary(wasm::BinaryOp::LtUInt32, a, b);
  } else {
    ICHECK(op->dtype.is_float());
    return builder_->makeBinary(wasm::BinaryOp::LtFloat32, a, b);
  }
}

wasm::Expression* CodeGenWasm::VisitExpr_(const LENode* op) {
  DLOG(INFO) << "Visit le node";
  wasm::Expression* a = MakeValue(op->a);
  wasm::Expression* b = MakeValue(op->b);
  if (op->dtype.is_int()) {
    return builder_->makeBinary(wasm::BinaryOp::LeSInt32, a, b);
  } else if (op->dtype.is_uint()) {
    return builder_->makeBinary(wasm::BinaryOp::LeUInt32, a, b);
  } else {
    ICHECK(op->dtype.is_float());
    return builder_->makeBinary(wasm::BinaryOp::LeFloat32, a, b);
  }
}

wasm::Expression* CodeGenWasm::VisitExpr_(const GTNode* op) {
  DLOG(INFO) << "Visit gt node";
  wasm::Expression* a = MakeValue(op->a);
  wasm::Expression* b = MakeValue(op->b);
  if (op->dtype.is_int()) {
    return builder_->makeBinary(wasm::BinaryOp::GtSInt32, a, b);
  } else if (op->dtype.is_uint()) {
    return builder_->makeBinary(wasm::BinaryOp::GtUInt32, a, b);
  } else {
    ICHECK(op->dtype.is_float());
    return builder_->makeBinary(wasm::BinaryOp::GtFloat32, a, b);
  }
}

wasm::Expression* CodeGenWasm::VisitExpr_(const GENode* op) {
  DLOG(INFO) << "Visit ge node";
  wasm::Expression* a = MakeValue(op->a);
  wasm::Expression* b = MakeValue(op->b);
  if (op->dtype.is_int()) {
    return builder_->makeBinary(wasm::BinaryOp::GeSInt32, a, b);
  } else if (op->dtype.is_uint()) {
    return builder_->makeBinary(wasm::BinaryOp::GeUInt32, a, b);
  } else {
    ICHECK(op->dtype.is_float());
    return builder_->makeBinary(wasm::BinaryOp::GeFloat32, a, b);
  }
}

wasm::Expression* CodeGenWasm::VisitExpr_(const EQNode* op) {
  DLOG(INFO) << "Visit eq node";
  wasm::Expression* a = MakeValue(op->a);
  wasm::Expression* b = MakeValue(op->b);
  if (op->dtype.is_int() || op->dtype.is_uint()) {
    return builder_->makeBinary(wasm::BinaryOp::EqInt32, a, b);
  } else {
    ICHECK(op->dtype.is_float());
    return builder_->makeBinary(wasm::BinaryOp::EqFloat32, a, b);
  }
}

wasm::Expression* CodeGenWasm::VisitExpr_(const NENode* op) {
  DLOG(INFO) << "Visit ne node";
  wasm::Expression* a = MakeValue(op->a);
  wasm::Expression* b = MakeValue(op->b);
  if (op->dtype.is_int() || op->dtype.is_uint()) {
    return builder_->makeBinary(wasm::BinaryOp::NeInt32, a, b);
  } else {
    ICHECK(op->dtype.is_float());
    return builder_->makeBinary(wasm::BinaryOp::NeFloat32, a, b);
  }
}

wasm::Expression* CodeGenWasm::VisitExpr_(const AndNode* op) {
  DLOG(INFO) << "Visit and node";
  ICHECK(op->dtype.is_int() || op->dtype.is_uint());
  wasm::Expression* a = MakeValue(op->a);
  wasm::Expression* b = MakeValue(op->b);
  return builder_->makeBinary(wasm::BinaryOp::AndInt32, a, b);
}

wasm::Expression* CodeGenWasm::VisitExpr_(const OrNode* op) {
  DLOG(INFO) << "Visit or node";
  ICHECK(op->dtype.is_int() || op->dtype.is_uint());
  wasm::Expression* a = MakeValue(op->a);
  wasm::Expression* b = MakeValue(op->b);
  return builder_->makeBinary(wasm::BinaryOp::OrInt32, a, b);
}

wasm::Expression* CodeGenWasm::VisitExpr_(const NotNode* op) {
  DLOG(INFO) << "Visit not node";
  LOG(FATAL) << "not implemented";
  return builder_->makeNop();
}

wasm::Expression* CodeGenWasm::VisitExpr_(const SelectNode* op) {
  DLOG(INFO) << "Visit select node";
  return builder_->makeSelect(MakeValue(op->condition), MakeValue(op->true_value),
                              MakeValue(op->false_value));
}

wasm::Expression* CodeGenWasm::VisitExpr_(const LetNode* op) {
  DLOG(INFO) << "Visit let node";
  wasm::Type wasm_dtype = DTypeToWasmType(op->dtype);
  wasm::NameType var_nt = CreateWasmVar(wasm_dtype);
  var_map_[op->var.get()] = var_nt;
  analyzer_->Bind(op->var, op->value);
  return MakeValue(op->body);
}

wasm::Expression* CodeGenWasm::VisitExpr_(const LoadNode* op) {
  LOG(FATAL) << "Unexpected deprecated LoadNode.  Use BufferLoadNode instead.";
  return nullptr;
}

wasm::Expression* CodeGenWasm::VisitExpr_(const BufferLoadNode* op) {
  return CreateBufferLoad(op, false);
}

wasm::Expression* CodeGenWasm::VisitExpr_(const CallNode* op) {
  DLOG(INFO) << "Visit call node";
  if (auto* ptr_op = op->op.as<OpNode>()) {
    DLOG(INFO) << "Op node";
    auto call_op = GetRef<Op>(ptr_op);
    if (op->op.same_as(builtin_call_extern_) || op->op.same_as(builtin_call_pure_extern_)) {
      // call extern intrinsic
      DLOG(INFO) << "Call extern intrinsic";
      ICHECK_GE(op->args.size(), 1U);
      auto global_symbol = Downcast<StringImm>(op->args[0]);
      //return this->CreateCallExtern(GetType(GetRef<PrimExpr>(op)), global_symbol->value, op->args,
      //                              true);
      LOG(FATAL) << "not implemented";
      return builder_->makeNop();
    } else if (op_attr_global_symbol_.count(call_op)) {
      // call extern if the op itself have a global symbol.
      DLOG(INFO) << "Call extern intrinsic (the op itself have a global symbol)";
      return this->CreateCallExtern(GetType(GetRef<PrimExpr>(op)), op_attr_global_symbol_[call_op],
                                    op->args, false);
    } else {
      DLOG(INFO) << "Create intrinsic: " << GetRef<Call>(op);
      auto x = CreateIntrinsic(op);
      return x;
    }
  } else {
    DLOG(INFO) << "Check node type";
    ICHECK(op->op.as<GlobalVarNode>());
    DLOG(INFO) << "Global var node";
    LOG(FATAL) << "Do not yet support cross function call";
    return builder_->makeNop();
  }
}

wasm::Expression* CodeGenWasm::VisitExpr_(const RampNode* op) {
  DLOG(INFO) << "Visit ramp node";
  DLOG(INFO) << "op_base " << op->base << ", op_stride " << op->stride
             << ", op_lanes " << op->lanes;
  
  LOG(FATAL) << "not implemented";
  return nullptr;
}

wasm::Expression* CodeGenWasm::VisitExpr_(const ShuffleNode* op) {
  LOG(FATAL) << "not implemented";
  return nullptr;
}

wasm::Expression* CodeGenWasm::VisitExpr_(const BroadcastNode* op) {
  DLOG(INFO) << "Visit broadcast node";
  DLOG(INFO) << "op_value " << op->value << ", op_lanes " << op->lanes;

  ICHECK_GE(op->lanes, 4) << "Broadcast node only supports 4 lanes";
  
#if defined(WASM_CODEGEN_COMBINE_INS)
  if (const BufferLoadNode* buffer_load = op->value.as<BufferLoadNode>()) {
    // NOTE(fucheng): Combine load and splat to achieve high performance.
    return CreateBufferLoad(buffer_load, true);
  } else if (const FloatImmNode* float_imm = op->value.as<FloatImmNode>()) {
    return CreateWasmFloatx4Const(float_imm->value);
  } else {
#endif
    wasm::Expression* value = MakeValue(op->value);
    const DataType dtype = op->dtype;
    if (dtype.is_int()) {
      return builder_->makeUnary(wasm::UnaryOp::SplatVecI32x4, value);
    } else if (dtype.is_float()) {
      return builder_->makeUnary(wasm::UnaryOp::SplatVecF32x4, value);
    } else {
      LOG(FATAL) << "not implemented";
      return nullptr;
    }
#if defined(WASM_CODEGEN_COMBINE_INS)
  }
#endif
}

wasm::Expression* CodeGenWasm::VisitStmt_(const StoreNode* op) {
  LOG(FATAL) << "Unexpected deprecated StoreNode.  Use BufferStoreNode instead.";
  return nullptr;
}

wasm::Expression* CodeGenWasm::VisitStmt_(const BufferStoreNode* op) {
  DLOG(INFO) << "Visit buffer store node";
  DataType value_dtype = op->value.dtype();
  DataType buffer_element_dtype = op->buffer->dtype;

  ICHECK_GE(op->indices.size(), 1)
      << "Buffer " << op->buffer->name << " is accessed with no indices.  "
      << "0-d scalar buffers are expected to be flattened to 1-d buffers prior to codegen.";
  
  for (size_t i = 0; i < op->indices.size() - 1; i++) {
    ICHECK_EQ(op->indices[i].dtype().lanes(), 1)
        << "Buffer " << op->buffer->name << " is accessed with a multi-lane index at position " << i
        << ".  Multi-lane indices are only supported as the last index.";
  }

  PrimExpr last_index = op->indices[op->indices.size() - 1];
  ICHECK_EQ(value_dtype.lanes(), last_index.dtype().lanes() * buffer_element_dtype.lanes());

  bool is_ramp_index = false;
  if (const RampNode* ramp_index = last_index.as<RampNode>()) {
    if (is_one(ramp_index->stride)) {
      last_index = ramp_index->base;
    }
    is_ramp_index = true;
  }

  int alignment;
  if (last_index.dtype().lanes() == 1) {
    int native_bits;
    GetAlignment(value_dtype, op->buffer->data.get(), last_index, &alignment, &native_bits);
  } else {
    ICHECK_GE(value_dtype.bits(), 8);
    alignment = value_dtype.bits() / 8;
  }

  wasm::Expression* last_index_value;
  const VarNode* offset_var = nullptr;
  const VarNode* base_index_var = nullptr;
  int64_t offset_value = 0;
  if (last_index.dtype().lanes() == 4) {
    if (const RampNode* ramp = last_index.as<RampNode>()) {
      // ICHECK(ramp->stride == 1);
      PrimExpr offset = ramp->base;
      last_index_value = MakeValue(offset);
    } else {
      last_index_value = MakeValue(last_index);
    }
  } else if (last_index.dtype().lanes() == 1) {
    last_index_value = MakeValue(last_index);
    if (const VarNode* var = last_index.as<VarNode>()) {
      if (IsCseVar(var)) {
        offset_var = var;
      }
      
      base_index_var = nullptr;
      GetCseVarOffset(offset_var, &base_index_var, &offset_value);
      if (base_index_var) {
        offset_var = base_index_var;
      }
    } else if (const AddNode* add = last_index.as<AddNode>()) {
      const VarNode* a_var = add->a.as<VarNode>();
      const IntImmNode* b_int_imm = add->b.as<IntImmNode>();
      if (a_var && b_int_imm) {
        if (IsCseVar(a_var)) {
          offset_var = a_var;
          offset_value = b_int_imm->value;
        }
      }
    }
  } else {
    ICHECK(false) << "Unsupported lanes " << last_index.dtype().lanes();
  }

  wasm::NameType wasm_offset_var;
  int32_t wasm_index_type = 0;
#if defined(WASM_CODEGEN_CONST_OFFSET)
  if (offset_var) {
    wasm_offset_var = var_map_[offset_var];
    if (IsCacheableCseVar(offset_var)) {
      wasm_index_type = 1;
      cached_cse_vars_.emplace_back(offset_var);
    } else if (IsCseVar(offset_var) && offset_value >= 0) {
      wasm_index_type = 2;
    }
  }
#endif

  wasm::NameType buf_var_nt = var_map_[op->buffer->data.get()];
  if (HasWasmBufferCachedVar(buf_var_nt)) {
    if (const IntImmNode* int_imm = last_index.as<IntImmNode>()) {
      wasm::NameType var_nt = GetWasmBufferCachedVar(buf_var_nt, int_imm->value);
      return builder_->makeLocalSet(GetWasmVarIndex(var_nt), MakeValue(op->value));
    }
  }

  int32_t addr_shift = 2;
  if (!is_ramp_index) {
    if (value_dtype.lanes() > 1) {
      addr_shift += std::log2(value_dtype.lanes());
    }
  }
  
  unsigned int bytes = static_cast<unsigned int>(GetTypeAllocSize(DTypeToWasmType(value_dtype)));
  wasm::Address offset = 0;
  unsigned int align = static_cast<unsigned int>(alignment);
  wasm::Expression* ptr_expr;
  wasm::Expression* value = MakeValue(op->value);
  wasm::Type type = DTypeToWasmType(value_dtype);
  if (wasm_index_type < 2) {
    wasm::Expression* ptr_offset = builder_->makeBinary(
        wasm::BinaryOp::ShlInt32, last_index_value, CreateWasmInt32Const(addr_shift));
    ptr_expr = builder_->makeBinary(
        wasm::BinaryOp::AddInt32, MakeValue(op->buffer->data), ptr_offset);
    if (wasm_index_type == 1) {
      ptr_expr = builder_->makeLocalTee(
          GetWasmVarIndex(wasm_offset_var), ptr_expr, wasm_offset_var.type);
    }
  } else if (wasm_index_type == 2) {
    ptr_expr = builder_->makeLocalGet(GetWasmVarIndex(wasm_offset_var), wasm_offset_var.type);
    offset = offset_value << addr_shift;
  }

  return builder_->makeStore(bytes, offset, align, ptr_expr, value, type, default_memory_name_);
}

wasm::Expression* CodeGenWasm::GetVarValue(const VarNode* v) const {
  DLOG(INFO) << "Get var value, name " << v->name_hint;
  auto it = var_map_.find(v);
  ICHECK(it != var_map_.end()) << "cannot find variable " << v->name_hint;
  wasm::NameType name_type = it->second;
  wasm::Index var_idx = GetWasmVarIndex(name_type);
  return builder_->makeLocalGet(var_idx, name_type.type);
}

bool CodeGenWasm::IsCseVar(const VarNode* v) const {
  const std::string prefix = "cse_var";
  const std::string tir_var_name = v->name_hint;
  if (tir_var_name.size() < prefix.size()) {
    return false;
  }
  return tir_var_name.substr(0, prefix.size()) == prefix;
}

bool CodeGenWasm::IsCacheableCseVar(const VarNode* v) const {
  if (!IsCseVar(v)) return false;
  for (auto it = cached_cse_vars_.begin();
      it != cached_cse_vars_.end(); it++) {
    if (*it == v) {
      return false;
    }
  }
  for (auto it = cse_var_offset_map_.begin();
      it != cse_var_offset_map_.end(); it++) {
    std::pair<const VarNode*, const VarNode*> cse_vars = it->first;
    DLOG(INFO) << "v " << v << ", second_v " << cse_vars.second;
    if (v == cse_vars.second) {
      return true;
    }
  }
  return true;
}

void CodeGenWasm::GetCseVarOffset(const VarNode* v, const VarNode** vbase, int64_t* offset) const {
  for (auto it = cached_cse_vars_.begin();
      it != cached_cse_vars_.end(); it++) {
    if (*it == v) {
      *vbase = *it;
      *offset = 0;
      return;
    }
  }
  for (auto it = cse_var_offset_map_.begin();
      it != cse_var_offset_map_.end(); it++) {
    std::pair<const VarNode*, const VarNode*> cse_vars = it->first;
    if (v == cse_vars.first) {
      *vbase = cse_vars.second;
      *offset = it->second;
      return;
    }
  }
  *vbase = nullptr;
  *offset = 0;
}

wasm::Expression* CodeGenWasm::CreateWasmInt32Const(int32_t value) {
  return builder_->makeConst(wasm::Literal(value));
}

wasm::Expression* CodeGenWasm::CreateWasmInt64Const(int64_t value) {
  return builder_->makeConst(wasm::Literal(value));
}

wasm::Expression* CodeGenWasm::CreateWasmFloatConst(float value) {
  return builder_->makeConst(wasm::Literal(value));
}

wasm::Expression* CodeGenWasm::CreateWasmFloatx4Const(float value) {
  wasm::Literal v(value);
  std::array<wasm::Literal, 4> arr;
  arr[0] = arr[1] = arr[2] = arr[3] = v;
  return builder_->makeConst(wasm::Literal(arr));
}

wasm::NameType CodeGenWasm::CreateWasmVar(const wasm::Type t) {
  size_t var_idx = wasm_params_.size() + wasm_vars_.size();
  wasm::NameType var_nt(wasm::NameType(wasm::Name::fromInt(var_idx), t));
  wasm_vars_.push_back(var_nt);
  DLOG(INFO) << "Create wasm var, type " << t << ", idx " << var_idx; 
  return var_nt;
}

void CodeGenWasm::CreateWasmCpuBuffer(const wasm::Type type,
                                      const uint64_t size,
                                      wasm::NameType* nt_ptr,
                                      wasm::Expression** expr_ptr) {
  wasm::Export* alloc_exp = module_->getExport(wasm::Name("TVMBackendAllocWorkspace"));
  ICHECK(alloc_exp->kind == wasm::ExternalKind::Function);
  wasm::Name alloc_func_name = alloc_exp->value;

  const int32_t dev_type_idx = 1;
  const int32_t dev_idx = 0;
  int32_t dtype_code;
  int32_t dtype_bits;
  WasmTypeToDLDataType(type, &dtype_code, &dtype_bits);
  
  std::vector<wasm::Expression*> arg_values;
  arg_values.push_back(CreateWasmInt32Const(dev_type_idx));
  arg_values.push_back(CreateWasmInt32Const(dev_idx));
  arg_values.push_back(CreateWasmInt64Const(static_cast<int64_t>(size)));
  arg_values.push_back(CreateWasmInt32Const(dtype_code));
  arg_values.push_back(CreateWasmInt32Const(dtype_bits));
  wasm::Expression* buf_ptr_expr = builder_->makeCall(alloc_func_name, arg_values, type, false);
  
  wasm::NameType nt = CreateWasmVar(wasm::Type::BasicType::i32);
  wasm::Index var_idx = GetWasmVarIndex(nt);
  wasm::Expression* out_expr = builder_->makeLocalSet(var_idx, buf_ptr_expr);
  
  *nt_ptr = nt;
  *expr_ptr = out_expr;
}

bool CodeGenWasm::IsLocalBufferVar(const VarNode* v) const {
  const std::string ending = "local";
  const std::string tir_var_name = v->name_hint;
  if (tir_var_name.length() < ending.length()) {
    return false;
  }
  return !tir_var_name.compare(tir_var_name.length() - ending.length(), ending.length(), ending);
}

bool CodeGenWasm::IsGlobalBufferVar(const VarNode* v) const {
  const std::string ending = "global";
  const std::string tir_var_name = v->name_hint;
  if (tir_var_name.length() < ending.length()) {
    return false;
  }
  return !tir_var_name.compare(tir_var_name.length() - ending.length(), ending.length(), ending);
}

void CodeGenWasm::CreateWasmCpuBufferCacheVars(const wasm::NameType buf_nt,
                                               const wasm::Type dtype,
                                               const int32_t constant_size) {
  ICHECK_LE(constant_size, 128) << "Can only allocate <= 128 local vars";
  wasm::NameType *buf_nt_ptr = new wasm::NameType(buf_nt);
  std::vector<wasm::NameType> *nts_ptr = new std::vector<wasm::NameType>();
  for (int32_t i = 0; i < constant_size; i++) {
    nts_ptr->push_back(CreateWasmVar(dtype));
  }
  buf_cache_var_map_[buf_nt_ptr] = nts_ptr;
}

bool CodeGenWasm::HasWasmBufferCachedVar(const wasm::NameType buf_nt) const {
  for (auto it = buf_cache_var_map_.begin();
      it != buf_cache_var_map_.end(); it++) {
    if (IsNameTypeSame(*(it->first), buf_nt)) {
      return true;
    }
  }
  return false;
}

wasm::NameType CodeGenWasm::GetWasmBufferCachedVar(const wasm::NameType buf_nt,
                                                   const int64_t idx) const {
  for (auto it = buf_cache_var_map_.begin();
      it != buf_cache_var_map_.end(); it++) {
    if (IsNameTypeSame(*(it->first), buf_nt)) {
      return it->second->at(idx);
    }
  }
  ICHECK(false) << "cannot find cached var for buffer " << buf_nt.name;
  return wasm::NameType("none", wasm::Type::BasicType::none);
}

wasm::Index CodeGenWasm::GetWasmVarIndex(wasm::NameType nt) const {
  for (auto it = wasm_params_.begin(); it != wasm_params_.end(); ++it) {
    if (IsNameTypeSame(*it, nt)) {
      wasm::Index idx = it - wasm_params_.begin();
      DLOG(INFO) << "Found in wasm params, idx " << idx;
      return idx;
    }
  }

  for (auto it = wasm_vars_.begin(); it != wasm_vars_.end(); ++it) {
    if (IsNameTypeSame(*it, nt)) {
      wasm::Index idx = wasm_params_.size() + (it - wasm_vars_.begin());
      DLOG(INFO) << "Found in wasm vars, idx " << idx;
      return idx;
    }
  }

  ICHECK(false) << "cannot find variable " << nt.name;
  return -1;
}

wasm::Expression* CodeGenWasm::CreateBufferLoad(const BufferLoadNode* op, bool is_broadcast) {
  DLOG(INFO) << "Visit buffer load node";
  DataType value_dtype = op->dtype;
  DataType buffer_element_dtype = op->buffer->dtype;
  
  ICHECK_GE(op->indices.size(), 1)
      << "Buffer " << op->buffer->name << " is accessed with no indices.  "
      << "0-d scalar buffers are expected to be flattened to 1-d buffers prior to codegen.";
  
  for (size_t i = 0; i < op->indices.size() - 1; i++) {
    ICHECK_EQ(op->indices[i].dtype().lanes(), 1)
        << "Buffer " << op->buffer->name << " is accessed with a multi-lane index at position " << i
        << ".  Multi-lane indices are only supported as the last index.";
  }

  PrimExpr last_index = op->indices[op->indices.size() - 1];
  ICHECK_EQ(value_dtype.lanes(), last_index.dtype().lanes() * buffer_element_dtype.lanes());

  bool is_ramp_index = false;
  if (const RampNode* ramp_index = last_index.as<RampNode>()) {
    if (is_one(ramp_index->stride)) {
      last_index = ramp_index->base;
    }
    is_ramp_index = true;
  }

  if (last_index.dtype().lanes() == 1 && HasAlignmentPadding(buffer_element_dtype)) {
    last_index = buffer_element_dtype.lanes() * last_index;
    buffer_element_dtype = buffer_element_dtype.element_of();
  }

  int alignment;
  if (last_index.dtype().lanes() == 1) {
    int native_bits;
    GetAlignment(value_dtype, op->buffer->data.get(), last_index, &alignment, &native_bits);
  } else {
    ICHECK_GE(value_dtype.bits(), 8);
    alignment = value_dtype.bits() / 8;
  }

  const VarNode* offset_var = nullptr;
  const VarNode* base_index_var = nullptr;
  int64_t index_offset = 0;
  wasm::Expression* last_index_value;
  if (last_index.dtype().lanes() == 4) {
    if (const RampNode* ramp = last_index.as<RampNode>()) {
      // ICHECK(ramp->stride == 1);
      PrimExpr offset = ramp->base;
      last_index_value = MakeValue(offset);
    } else {
      ICHECK(false) << "We only support 4 lanes for ramp node";
      //last_index_value = MakeValue(last_index);
    }
  } else if (last_index.dtype().lanes() == 1) {
    last_index_value = MakeValue(last_index);
    if (const VarNode* var = last_index.as<VarNode>()) {
      offset_var = var;

      base_index_var = nullptr;
      GetCseVarOffset(offset_var, &base_index_var, &index_offset);
      if (base_index_var) {
        offset_var = base_index_var;
      }
    } else if (const AddNode* add = last_index.as<AddNode>()) {
      const VarNode* a_var = add->a.as<VarNode>();
      const IntImmNode* b_int_imm = add->b.as<IntImmNode>();
      if (a_var && b_int_imm) {
        if (IsCseVar(a_var)) {
          offset_var = a_var;
          index_offset = b_int_imm->value;
        }
      }
    }
  } else {
    ICHECK(false) << "Unsupported lanes " << last_index.dtype().lanes();
  }

  DLOG(INFO) << "Check cse var";
  /**
   * Wasm Index Type
   *   0 = Normal
   *   1 = Cacheable
   *   2 = Replaceable
   */
  int32_t wasm_index_type = 0;
  wasm::NameType wasm_offset_var;
#if defined(WASM_CODEGEN_CONST_OFFSET)
  if (offset_var) {
    wasm_offset_var = var_map_[offset_var];
    if (IsCacheableCseVar(offset_var)) {
      wasm_index_type = 1;
      cached_cse_vars_.emplace_back(offset_var);
    } else if (IsCseVar(offset_var) && index_offset >= 0) {
      wasm_index_type = 2;
    }
  }
#endif

  DLOG(INFO) << "lanes " << value_dtype.lanes()
             << ", alignment " << alignment
             << ", op_buffer_data " << op->buffer->data;
  DLOG(INFO) << "wasm_index_type " << wasm_index_type
             << ", index_offset " << index_offset;

  wasm::NameType buf_var_nt = var_map_[op->buffer->data.get()];
  if (HasWasmBufferCachedVar(buf_var_nt)) {
    if (const IntImmNode* int_imm = last_index.as<IntImmNode>()) {
      wasm::NameType var_nt = GetWasmBufferCachedVar(buf_var_nt, int_imm->value);
      return builder_->makeLocalGet(GetWasmVarIndex(var_nt), var_nt.type);
    }
  }

  int32_t addr_shift = 2;
  if (!is_ramp_index) {
    if (value_dtype.lanes() > 1) {
      addr_shift += std::log2(value_dtype.lanes());
    }
  }

  unsigned int bytes = static_cast<unsigned int>(GetTypeAllocSize(DTypeToWasmType(value_dtype)));
  bool is_signed = false;
  wasm::Address offset = 0;
  unsigned int align = static_cast<unsigned int>(alignment);
  wasm::Expression* ptr_expr = nullptr;
  if (wasm_index_type < 2) {
    wasm::Expression* ptr_offset = builder_->makeBinary(
        wasm::BinaryOp::ShlInt32, last_index_value, CreateWasmInt32Const(addr_shift));
    ptr_expr = builder_->makeBinary(
      wasm::BinaryOp::AddInt32, MakeValue(op->buffer->data), ptr_offset);
    if (wasm_index_type == 1) {
      ptr_expr = builder_->makeLocalTee(
          GetWasmVarIndex(wasm_offset_var), ptr_expr, wasm_offset_var.type);
    }
  } else if (wasm_index_type == 2) {
    ptr_expr = builder_->makeLocalGet(GetWasmVarIndex(wasm_offset_var), wasm_offset_var.type);
    offset = index_offset << addr_shift;
  } else {
    ICHECK(false) << "Unsupported wasm index type " << wasm_index_type;
  }
  wasm::Type type = DTypeToWasmType(value_dtype);

  wasm::Expression* load_expr = nullptr;
  if (!is_broadcast) {
    load_expr = builder_->makeLoad(
        bytes,
        is_signed,
        offset,
        align,
        ptr_expr,
        type,
        default_memory_name_);
  } else {
    load_expr = builder_->makeSIMDLoad(
        wasm::SIMDLoadOp::Load32SplatVec128,
        offset,
        align,
        ptr_expr,
        default_memory_name_);
    type = wasm::Type::BasicType::v128;
  }
  wasm::Expression* out_expr = load_expr;

#if defined(WASM_CODEGEN_BUF_TO_VAR)
  const VarNode* data_var = op->buffer->data.as<VarNode>();
  if (data_var && offset_var) {
    std::pair<const VarNode*, const VarNode*> data_offset_key(data_var, offset_var);
    std::pair<std::pair<const VarNode*, const VarNode*>, int64_t> load_key(data_offset_key, index_offset);
    if (load_cache_var_map_.count(load_key)) {
      wasm::NameType cache_wasm_var = load_cache_var_map_[load_key];
      out_expr = builder_->makeLocalGet(
          GetWasmVarIndex(cache_wasm_var), cache_wasm_var.type);
    } else {
      wasm::NameType cache_wasm_var = CreateWasmVar(type);
      load_cache_var_map_[load_key] = cache_wasm_var;
      out_expr = builder_->makeLocalTee(
          GetWasmVarIndex(cache_wasm_var), load_expr, cache_wasm_var.type);
    }
  }
#endif

  return out_expr;
}

wasm::Expression* CodeGenWasm::CreateSerialFor(
    wasm::Expression* begin,
    wasm::Expression* end,
    wasm::Expression* stride,
    const Var& loop_var,
    const Stmt& body) {
  wasm::Type loop_value_type = wasm::Type(wasm::Type::BasicType::i32);
  wasm::NameType loop_value_name_type = CreateWasmVar(loop_value_type);
  var_map_[loop_var.get()] = loop_value_name_type;
  wasm::Index loop_value_index = GetWasmVarIndex(loop_value_name_type);
  wasm::Expression* init_loop_value = builder_->makeLocalSet(loop_value_index, begin);
  wasm::Expression* loop_value = builder_->makeLocalGet(loop_value_index, loop_value_type);

  stmt_body_exprs_.push_back(std::vector<wasm::Expression*>());

  wasm::Expression* body_block;
  if (stmt_body_exprs_.size() <= 1000) {
    body_block = this->VisitStmt(body);
  } else {
    body_block = builder_->makeBlock(builder_->makeNop());
  }
  
  wasm::Expression* lt = builder_->makeBinary(wasm::BinaryOp::LtSInt32, loop_value, end);
  wasm::Expression* br_if = builder_->makeBreak(wasm::Name(loop_var->name_hint.c_str()), NULL, lt);
  wasm::Expression* add_loop_value = builder_->makeLocalSet(
      loop_value_index, builder_->makeBinary(wasm::BinaryOp::AddInt32, loop_value, stride));

  //std::vector<wasm::Expression*> loop_exprs = {body_block, add_loop_value, br_if};

  ICHECK_GT(stmt_body_exprs_.size(), 1);
  stmt_body_exprs_.back().push_back(body_block);
  stmt_body_exprs_.back().push_back(add_loop_value);
  stmt_body_exprs_.back().push_back(br_if);
  std::vector<wasm::Expression*> loop_exprs = stmt_body_exprs_.back();
  stmt_body_exprs_.pop_back();

  DLOG(INFO) << "loop_exprs_size " << loop_exprs.size();
  
  wasm::Expression* loop_blk = builder_->makeBlock(loop_exprs);
  wasm::Expression* loop = builder_->makeLoop(wasm::Name(loop_var->name_hint.c_str()), loop_blk);

  std::vector<wasm::Expression*> total_loop_exprs = {init_loop_value, loop};
  return builder_->makeBlock(total_loop_exprs);
}

wasm::Expression* CodeGenWasm::VisitStmt_(const ForNode* op) {
  DLOG(INFO) << "Visit for node";
  ICHECK(is_zero(op->min));
  analyzer_->Bind(op->loop_var, Range::FromMinExtent(op->min, op->extent));
  if (op->kind == ForKind::kUnrolled) {
    LOG(WARNING) << "Unroll hint get ignore at CodeGenWasm backend, "
                 << " consider set unroll_explicit=True";
  } else {
    ICHECK(op->kind == ForKind::kSerial);
  }
  return CreateSerialFor(MakeValue(op->min), MakeValue(op->extent),
                         CreateWasmInt32Const(1),
                         op->loop_var, op->body);
}

wasm::Expression* CodeGenWasm::VisitStmt_(const WhileNode* op) {
  LOG(FATAL) << "not implemented";
  return nullptr;
}

wasm::Expression* CodeGenWasm::VisitStmt_(const IfThenElseNode* op) {
  DLOG(INFO) << "Visit if then else node";
  DLOG(INFO) << "Make condition";
  if (op->condition.as<NotNode>() || op->condition.as<CallNode>()) {
    return builder_->makeNop();
  }
  if (const NENode* nen = op->condition.as<NENode>()) {
    if (const CallNode* call = nen->a.as<CallNode>()) {
      return MakeValue(nen->a);
    }
  }
  wasm::Expression* cond = MakeValue(op->condition);
  DLOG(INFO) << "op_else_case_defined " << op->else_case.defined();
  if (op->else_case.defined()) {
    wasm::Expression* then_block = this->VisitStmt(op->then_case);
    wasm::Expression* else_block = this->VisitStmt(op->else_case);
    return builder_->makeIf(cond, then_block, else_block);
  } else {
    wasm::Expression* then_block = this->VisitStmt(op->then_case);
    return builder_->makeIf(cond, then_block);
  }
}

wasm::Expression* CodeGenWasm::VisitStmt_(const AllocateNode* op) {
  DLOG(INFO) << "Visit allocate node";
  ICHECK_EQ(op->extents.size(), 1)
      << "WASM codegen only supports flat 1-d buffer allocation, but allocation of "
      << op->buffer_var->name_hint << " is " << op->extents << "-d";
  
  ICHECK(!is_zero(op->condition));

  int32_t constant_size = op->ConstantAllocationSize();
  ICHECK_GT(constant_size, 0) << "Can only handle constant size stack allocation";
  //StorageInfo& info = alloc_storage_info_[op->buffer_var.get()];

  DLOG(INFO) << "Create CPU buffer, dtype " << op->dtype << ", constant_size " << constant_size;
  
  const uint64_t dsize = constant_size * (op->dtype.bits() * op->dtype.lanes() / 8);
  wasm::NameType buf;
  wasm::Expression* alloc_expr;
  CreateWasmCpuBuffer(DTypeToWasmType(op->dtype), dsize, &buf, &alloc_expr);

  ICHECK(!var_map_.count(op->buffer_var.get()));
  var_map_[op->buffer_var.get()] = buf;

#if defined(WASM_CODEGEN_BUF_TO_VAR)
  if (IsLocalBufferVar(op->buffer_var.get())) {
    CreateWasmCpuBufferCacheVars(buf, DTypeToWasmType(op->dtype), constant_size);
  }
#endif

  wasm::Expression* body_expr = this->VisitStmt(op->body);

  std::vector<wasm::Expression*> exprs = {alloc_expr, body_expr};
  return builder_->makeBlock(exprs);
}

wasm::Expression* CodeGenWasm::VisitStmt_(const AllocateConstNode* op) {
  DLOG(INFO) << "Visit allocate const node";
  return this->VisitStmt(op->body);
}

wasm::Expression* CodeGenWasm::VisitStmt_(const AttrStmtNode* op) {
  DLOG(INFO) << "Visit attr stmt node";
  DLOG(INFO) << "op_attr_key " << op->attr_key;
  return this->VisitStmt(op->body);
}

wasm::Expression* CodeGenWasm::VisitStmt_(const AssertStmtNode* op) {
  DLOG(INFO) << "Visit assert stmt node";
  With<arith::ConstraintContext> cctx(analyzer_.get(), op->condition);
  return this->VisitStmt(op->body);
}

wasm::Expression* CodeGenWasm::VisitStmt_(const LetStmtNode* op) {
  DLOG(INFO) << "Visit let stmt node";
  const VarNode* v = op->var.get();
  ICHECK(!var_map_.count(v));
  DLOG(INFO) << "name_hint " << v->name_hint;

  wasm::Expression* local_set_expr = nullptr;
  if (IsLocalBufferVar(v) && !enable_buf_to_var_) {
    enable_buf_to_var_ = true;
    //local_set_expr = builder_->makeNop();
  }

  wasm::NameType var_name_type = CreateWasmVar(DTypeToWasmType(v->dtype));
  last_wasm_var_ = var_name_type;
  wasm::Index var_idx = GetWasmVarIndex(var_name_type);
  wasm::Expression* var_value = MakeValue(op->value);
  if (local_set_expr == nullptr) {
    local_set_expr = builder_->makeLocalSet(var_idx, var_value);
  }
  //if (stmt_body_exprs_.size() > 0) {
  //  stmt_body_exprs_.back().push_back(local_set_expr);
  //}
  var_map_[v] = var_name_type;
  analyzer_->Bind(op->var, op->value);

  // NOTE(fucheng): Cached cse var offset.
  if (IsCseVar(v)) {
    if (const AddNode* add = op->value.as<AddNode>()) {
      if (const VarNode* va = add->a.as<VarNode>()) {
        if (IsCseVar(va)) {
          if (const IntImmNode* vb = add->b.as<IntImmNode>()) {
            std::pair csr_var_pair(v, va);
            cse_var_offset_map_[csr_var_pair] = vb->value;
            DLOG(INFO) << "v_name " << v->name_hint << ", va_name " << va->name_hint << ", offset " << vb->value;
          }
        }
      }
    }
  }

  wasm::Expression* body_expr = this->VisitStmt(op->body);

  std::vector<wasm::Expression*> exprs;
  exprs.push_back(local_set_expr);
  exprs.push_back(body_expr);
  return builder_->makeBlock(exprs);
}

wasm::Expression* CodeGenWasm::VisitStmt_(const SeqStmtNode* op) {
  DLOG(INFO) << "Visit seq stmt node";
  seq_stmt_depth_ += 1;
  DLOG(INFO) << "seq_stmt_depth " << seq_stmt_depth_;
  std::vector<wasm::Expression*> stmt_exprs;
  for (Stmt stmt : op->seq) {
    if (seq_stmt_depth_ < 1000) {
      DLOG(INFO) << "New stmt start";
      wasm::Expression* expr = this->VisitStmt(stmt);
      stmt_exprs.push_back(expr);
      //stmt_exprs.push_back(builder_->makeNop());
      DLOG(INFO) << "New stmt end";
    }
  }
  DLOG(INFO) << "stmt_exprs_size " << stmt_exprs.size();
  //if (stmt_exprs.size() > 2) {
  //  return builder_->makeNop();
  //}
  seq_stmt_depth_ -= 1;
  return builder_->makeBlock(stmt_exprs);
}

wasm::Expression* CodeGenWasm::VisitStmt_(const EvaluateNode* op) {
  DLOG(INFO) << "Visit evaluate node";
  return MakeValue(op->value);
}

wasm::Expression* CodeGenWasm::CreateIntrinsic(const CallNode* op) {
  DataType t = op->dtype;
  if (op->op.same_as(builtin::tvm_struct_get())) {
    ICHECK_EQ(op->args.size(), 3U);
    int kind = op->args[2].as<IntImmNode>()->value;
    if (kind == builtin::kArrAddr) {
      wasm::Expression* ptr = MakeValue(op->args[0]);
      wasm::Expression* index = MakeValue(op->args[1]);
      return builder_->makeBinary(wasm::BinaryOp::AddInt32, ptr, index);
    } else if (kind == builtin::kArrData) {
      unsigned int bytes = static_cast<unsigned int>(GetTypeAllocSize(DTypeToWasmType(t)));
      bool is_signed = false;
      wasm::Address offset = 0;
      unsigned int align = 0;
      wasm::Expression* ptr = MakeValue(op->args[0]);
      wasm::Type type = DTypeToWasmType(t);
      return builder_->makeLoad(
          bytes,
          is_signed,
          offset,
          align,
          ptr,
          type,
          default_memory_name_);
    } else if (kind == builtin::kTVMValueContent) {
      ICHECK_EQ(t.lanes(), 1);
      ICHECK(t.is_handle() || t.bits() == 64);
      unsigned int bytes = static_cast<unsigned int>(GetTypeAllocSize(DTypeToWasmType(t)));
      bool is_signed = false;
      //wasm::Address offset = 0;
      wasm::Address offset = op->args[1].as<IntImmNode>()->value * 8;
      unsigned int align = 0;
      //wasm::Expression* ptr = builder_->makeBinary(
      //    wasm::BinaryOp::AddInt32, MakeValue(op->args[0]),
      //        builder_->makeBinary(wasm::BinaryOp::MulInt32,
      //                              MakeValue(op->args[1]),
      //                              CreateWasmInt32Const(8)));
      wasm::Expression* ptr = MakeValue(op->args[0]);
      wasm::Type type = DTypeToWasmType(t);
      return builder_->makeLoad(
          bytes,
          is_signed,
          offset,
          align,
          ptr,
          type,
          default_memory_name_);
    } else if (kind == builtin::kArrDeviceId) {
      unsigned int bytes = 4;
      bool is_signed = false;
      wasm::Address offset = 2 * bytes;
      unsigned int align = 0;
      wasm::Expression* ptr = MakeValue(op->args[0]);
      wasm::Type type = DTypeToWasmType(t);
      
      return builder_->makeLoad(
          bytes,
          is_signed,
          offset,
          align,
          ptr,
          type,
          default_memory_name_);
    } else {
      unsigned int bytes = static_cast<unsigned int>(GetTypeAllocSize(DTypeToWasmType(t)));
      bool is_signed = false;
      wasm::Address offset = 0;
      unsigned int align = 0;
      wasm::Expression* ptr = builder_->makeBinary(
          wasm::BinaryOp::AddInt32, MakeValue(op->args[0]), MakeValue(op->args[1]));
      wasm::Type type = DTypeToWasmType(t);
      return builder_->makeLoad(
          bytes,
          is_signed,
          offset,
          align,
          ptr,
          type,
          default_memory_name_);
    }
  } else if (op->op.same_as(builtin::bitwise_and())) {
    return builder_->makeBinary(wasm::BinaryOp::AndInt32, MakeValue(op->args[0]), MakeValue(op->args[1]));
  } else if (op->op.same_as(builtin::shift_right())) {
    if (op->args[0].dtype().is_int()) {
      return builder_->makeBinary(wasm::BinaryOp::ShrSInt32, MakeValue(op->args[0]), MakeValue(op->args[1]));
    } else {
      return builder_->makeBinary(wasm::BinaryOp::ShrUInt32, MakeValue(op->args[0]), MakeValue(op->args[1]));
    }
  } else if (op->op.same_as(builtin::if_then_else())) {
    ICHECK_EQ(op->args[0].dtype().lanes(), 1) << "if_then_else can only take scalar condition";
    wasm::Expression* cond = MakeValue(op->args[0]);
    wasm::Expression* then_block = MakeValue(op->args[1]);
    wasm::Expression* else_block = MakeValue(op->args[2]);
    return builder_->makeIf(cond, then_block, else_block);
  } else {
    //LOG(FATAL) << "not implemented";
    return builder_->makeNop();
  }
}

int CodeGenWasm::NativeVectorBits(const runtime::StorageScope& storage_scope) const {
  return native_vector_bits_;
}

void CodeGenWasm::Optimize() {
  /**
  wasm::PassOptions pass_options = wasm::PassOptions::getWithDefaultOptimizationOptions();
  pass_options.lowMemoryUnused = true;
  wasm::PassRunner pass_runner(module_.get());
  pass_runner.options = pass_options;
  pass_runner.add("local-cse");
  pass_runner.add("optimize-added-constants");
  pass_runner.add("optimize-added-constants-propagate");
  pass_runner.run();
  */
}

int32_t CodeGenWasm::GetWasmTypeBytes(const wasm::Type type) const {
  if (type == wasm::Type::BasicType::i32 || type == wasm::Type::BasicType::f32) {
    return 4;
  } else if (type == wasm::Type::BasicType::i64 || type == wasm::Type::BasicType::f64) {
    return 8;
  } else if (type == wasm::Type::BasicType::v128) {
    return 16;
  } else {
    LOG(FATAL) << "Unknown wasm data type " << type;
    return -1;
  }
}

wasm::Type CodeGenWasm::DTypeToWasmType(const DataType& dtype) const {
  if (dtype.is_handle()) {
    ICHECK_EQ(dtype.lanes(), 1);
    return t_void_p_;
  }
  if (dtype.is_void()) {
    return t_void_;
  }
  wasm::Type etype = wasm::Type::BasicType::none;
  if (dtype.is_int() || dtype.is_uint()) {
    switch (dtype.bits()) {
      case 32:
        etype = wasm::Type::BasicType::i32;
        break;
      case 64:
        etype = wasm::Type::BasicType::i64;
        break;
      default:
        LOG(FATAL) << "do not support " << dtype;
    }
  } else if (dtype.is_float()) {
    switch (dtype.bits()) {
      case 32:
        etype = wasm::Type::BasicType::f32;
        break;
      case 64:
        etype = wasm::Type::BasicType::f64;
        break;
      default:
        LOG(FATAL) << "do not support " << dtype;
    }
  }
  if (dtype.lanes() == 4) {
    etype = wasm::Type::BasicType::v128;
  }

  if (etype == wasm::Type::BasicType::none) {
    LOG(FATAL) << "Unknown data type " << dtype;
  }

  return etype;
}

wasm::Type CodeGenWasm::GetWasmType(const Type& type) const {
  if (auto* ptr = type.as<PrimTypeNode>()) {
    return DTypeToWasmType(ptr->dtype);
  } else if (auto* ptr = type.as<PointerTypeNode>()) {
    if (auto* primtype = ptr->element_type.as<PrimTypeNode>()) {
      if (primtype->dtype.is_void()) {
        return t_void_p_;
      }
    }
    return GetWasmType(ptr->element_type);
  } else if (IsVoidType(type)) {
    return t_void_;
  } else {
    LOG(FATAL) << "Type " << type << " does not have a corresponding Wasm Type";
    return t_void_;
  }
}

wasm::Type CodeGenWasm::GetWasmType(const PrimExpr& expr) const {
  return GetWasmType(GetType(expr));
}

bool CodeGenWasm::IsNameTypeSame(const wasm::NameType nt_a, const wasm::NameType nt_b) const {
  return nt_a.name.equals(nt_b.name.toString()) && (nt_a.type == nt_b.type);
}

int CodeGenWasm::GetTypeAllocSize(wasm::Type type) {
  return static_cast<int>(type.getByteSize());
}

bool CodeGenWasm::HasAlignmentPadding(DataType dtype) {
  int bytes = GetTypeAllocSize(DTypeToWasmType(dtype));
  int bytes_scalar = GetTypeAllocSize(DTypeToWasmType(dtype.element_of()));
  return bytes != bytes_scalar * dtype.lanes();
}

void CodeGenWasm::GetAlignment(DataType t, const VarNode* buf_var, const PrimExpr& index,
                               int* p_alignment, int* p_native_bits) {
  int max_align_bits = t.bits();
  auto it = alloc_storage_info_.find(buf_var);
  if (it != alloc_storage_info_.end()) {
    const StorageInfo& info = it->second;
    *p_native_bits =
        NativeVectorBits(runtime::StorageScope::Create(GetPtrStorageScope(GetRef<Var>(buf_var))));
    max_align_bits = info.alignment * 8;
  } else {
    *p_native_bits = native_vector_bits_;
  }

  arith::ModularSet me = analyzer_->modular_set(index);
  int64_t base = me->base;
  int64_t coeff = me->coeff;

  int align_bits = t.bits();
  while (align_bits < max_align_bits && base % 2 == 0 && coeff % 2 == 0) {
    base = base / 2;
    coeff = coeff / 2;
    align_bits *= 2;
  }
  if (align_bits < 8) {
    align_bits = 8;
  }
  *p_alignment = align_bits / 8;
}

wasm::Expression* CodeGenWasm::CreateCallExtern(Type ret_type, String global_symbol,
                                                const Array<PrimExpr>& args,
                                                bool skip_first_arg) {
  DLOG(INFO) << "Create call extern";
  wasm::Export* exp = module_->getExport(wasm::Name(global_symbol.c_str()));
  ICHECK(exp->kind == wasm::ExternalKind::Function);
  wasm::Name internal_func_name = exp->value;

  std::vector<wasm::Expression*> arg_values;
  for (size_t i = static_cast<size_t>(skip_first_arg); i < args.size(); ++i) {
    arg_values.push_back(MakeValue(args[i]));
  }

  wasm::Type type = GetWasmType(ret_type);

  DLOG(INFO) << "internal_func_name " << internal_func_name << ", ret_type " << ret_type;

#if defined(WASM_CODEGEN_BUF_TO_VAR)
  if ((!global_symbol.compare("TVMBackendAllocWorkspace")
      || !global_symbol.compare("legalstub$TVMBackendAllocWorkspace"))
      && enable_buf_to_var_) {
    DLOG(INFO) << "last_wasm_var " << last_wasm_var_.name;
    wasm::Type wasm_type = wasm::Type::BasicType::v128;
    const int32_t constant_size = args[2].as<IntImmNode>()->value / GetWasmTypeBytes(wasm_type);
    CreateWasmCpuBufferCacheVars(last_wasm_var_, wasm_type, constant_size);
  }
#endif

  return builder_->makeCall(internal_func_name, arg_values, type, false);
}

void CodeGenWasm::WasmTypeToDLDataType(wasm::Type type,
                                       int32_t* dtype_code_ptr,
                                       int32_t* dtype_bits_ptr) {
  if (type == wasm::Type::BasicType::i32) {
    *dtype_code_ptr = 0;
    *dtype_bits_ptr = 32;
  } else if (type == wasm::Type::BasicType::i64) {
    *dtype_code_ptr = 0;
    *dtype_bits_ptr = 64;
  } else if (type == wasm::Type::BasicType::f32) {
    *dtype_code_ptr = 2;
    *dtype_bits_ptr = 32;
  } else if (type == wasm::Type::BasicType::f64) {
    *dtype_code_ptr = 2;
    *dtype_bits_ptr = 64;
  } else if (type == wasm::Type::BasicType::v128) {
    // NOTE(fucheng): We treat it as floatx4.
    *dtype_code_ptr = 2;
    *dtype_bits_ptr = 32;
  } else {
    LOG(FATAL) << "do not support " << type;
  }
}

void CodeGenWasm::AddWasmExportedMemory() {
  auto memory = std::make_unique<wasm::Memory>();
  memory->name = default_memory_name_;
  memory->initial = default_initial_memory_size_;
  memory->max = default_max_memory_size_;
  memory->shared = false;
  memory->indexType = wasm::Type::i32;

  auto memoryExport = std::make_unique<wasm::Export>();
  memoryExport->name = "memory";
  memoryExport->value = memory->name;
  memoryExport->kind = wasm::ExternalKind::Memory;
  module_->addExport(memoryExport.release());

  module_->addMemory(std::move(memory));
}

}  // namespace codegen
}  // namespace tvm
