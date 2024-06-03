
#ifndef TVM_TARGET_WASM_CODEGEN_WASM_H_
#define TVM_TARGET_WASM_CODEGEN_WASM_H_

#include <wasm-builder.h>

#include <tvm/arith/analyzer.h>
#include <tvm/ir/module.h>
#include <tvm/target/codegen.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

#include "../../runtime/thread_storage_scope.h"
#include "../../tir/transforms/ir_utils.h"

#define WASM_CODEGEN_BUF_TO_VAR
#define WASM_CODEGEN_CONST_OFFSET
#define WASM_CODEGEN_COMBINE_INS

namespace tvm {
namespace codegen {

using namespace tir;

class CodeGenWasm : public ExprFunctor<wasm::Expression*(const PrimExpr&)>,
                    public StmtFunctor<wasm::Expression*(const Stmt&)> {
 public:
  CodeGenWasm();
  ~CodeGenWasm();

  void Init();

  void AddFunction(const PrimFunc& f);

  void AddMainFunction(const std::string& entry_func_name);

  std::unique_ptr<wasm::Module> Finish();

  template <typename IterType, typename ConvType>
  void AddFunctionsOrdered(IterType begin, IterType end, ConvType pfunc);

  template <typename IterType>
  void AddFunctionsOrdered(IterType begin, IterType end) {
    this->AddFunctionsOrdered(begin, end, [](auto f) { return f; });
  }

  wasm::Expression* MakeValue(const PrimExpr& e) { return VisitExpr(e); }

  wasm::Expression* VisitExpr_(const VarNode* op) override;
  wasm::Expression* VisitExpr_(const CastNode* op) override;
  wasm::Expression* VisitExpr_(const IntImmNode* op) override;
  wasm::Expression* VisitExpr_(const FloatImmNode* op) override;
  wasm::Expression* VisitExpr_(const StringImmNode* op) override;
  wasm::Expression* VisitExpr_(const AddNode* op) override;
  wasm::Expression* VisitExpr_(const SubNode* op) override;
  wasm::Expression* VisitExpr_(const MulNode* op) override;
  wasm::Expression* VisitExpr_(const DivNode* op) override;
  wasm::Expression* VisitExpr_(const ModNode* op) override;
  wasm::Expression* VisitExpr_(const MinNode* op) override;
  wasm::Expression* VisitExpr_(const MaxNode* op) override;
  wasm::Expression* VisitExpr_(const LTNode* op) override;
  wasm::Expression* VisitExpr_(const LENode* op) override;
  wasm::Expression* VisitExpr_(const GTNode* op) override;
  wasm::Expression* VisitExpr_(const GENode* op) override;
  wasm::Expression* VisitExpr_(const EQNode* op) override;
  wasm::Expression* VisitExpr_(const NENode* op) override;
  wasm::Expression* VisitExpr_(const AndNode* op) override;
  wasm::Expression* VisitExpr_(const OrNode* op) override;
  wasm::Expression* VisitExpr_(const NotNode* op) override;
  wasm::Expression* VisitExpr_(const SelectNode* op) override;
  wasm::Expression* VisitExpr_(const LetNode* op) override;
  wasm::Expression* VisitExpr_(const BufferLoadNode* op) override;
  wasm::Expression* VisitExpr_(const CallNode* op) override;
  wasm::Expression* VisitExpr_(const RampNode* op) override;
  wasm::Expression* VisitExpr_(const ShuffleNode* op) override;
  wasm::Expression* VisitExpr_(const BroadcastNode* op) override;

  wasm::Expression* VisitStmt_(const BufferStoreNode* op) override;
  wasm::Expression* VisitStmt_(const ForNode* op) override;
  wasm::Expression* VisitStmt_(const WhileNode* op) override;
  wasm::Expression* VisitStmt_(const IfThenElseNode* op) override;
  wasm::Expression* VisitStmt_(const AllocateNode* op) override;
  wasm::Expression* VisitStmt_(const AllocateConstNode* op) override;
  wasm::Expression* VisitStmt_(const AttrStmtNode* op) override;
  wasm::Expression* VisitStmt_(const AssertStmtNode* op) override;
  wasm::Expression* VisitStmt_(const LetStmtNode* op) override;
  wasm::Expression* VisitStmt_(const SeqStmtNode* op) override;
  wasm::Expression* VisitStmt_(const EvaluateNode* op) override;
  wasm::Expression* VisitStmt_(const DeclBufferNode* op) override;

  wasm::Expression* CreateIntrinsic(const CallNode* op);

 protected:
  struct StorageInfo {
    int alignment{0};
  };

  std::string GetSymbolName(const GlobalVar& gvar, const PrimFunc& func);

  virtual int NativeVectorBits(const runtime::StorageScope& storage_scope) const;

  void Optimize();

  int32_t GetWasmTypeBytes(const wasm::Type type) const;

  wasm::Type DTypeToWasmType(const DataType& dtype) const;

  wasm::Type GetWasmType(const Type& type) const;

  wasm::Type GetWasmType(const PrimExpr& expr) const;

  bool IsNameTypeSame(const wasm::NameType nt_a, const wasm::NameType nt_b) const;

  int GetTypeAllocSize(wasm::Type type);

  bool HasAlignmentPadding(DataType dtype);

  void GetAlignment(DataType t, const VarNode* buf_var, const PrimExpr& index, int* p_alignment,
                    int* p_native_bits);

  wasm::Expression* GetVarValue(const VarNode* v) const;

  bool IsCseVar(const VarNode* v) const;

  bool IsCacheableCseVar(const VarNode* v) const;

  void GetCseVarOffset(const VarNode* v, const VarNode** vbase, int64_t* offset) const;
  
  wasm::Expression* CreateWasmInt32Const(int32_t value);

  wasm::Expression* CreateWasmInt64Const(int64_t value);
  
  wasm::Expression* CreateWasmFloatConst(float value);

  wasm::Expression* CreateWasmFloatx4Const(float value);

  wasm::NameType CreateWasmVar(const wasm::Type t);

  void CreateWasmCpuBuffer(const wasm::Type t,
                           const uint64_t size,
                           wasm::NameType* nt_ptr,
                           wasm::Expression** expr_ptr);

  bool IsLocalBufferVar(const VarNode* v) const;

  bool IsGlobalBufferVar(const VarNode* v) const;

  void CreateWasmCpuBufferCacheVars(const wasm::NameType buf_nt,
                                    const wasm::Type dtype,
                                    const int32_t constant_size);

  bool HasWasmBufferCachedVar(const wasm::NameType buf_nt) const;

  wasm::NameType GetWasmBufferCachedVar(const wasm::NameType buf_nt, const int64_t idx) const;

  wasm::Index GetWasmVarIndex(wasm::NameType nt) const;

  wasm::Expression* CreateBufferLoad(const BufferLoadNode* op, bool is_broadcast);
  
  wasm::Expression* CreateAdd(DataType t, wasm::Expression* a, wasm::Expression* b);
  wasm::Expression* CreateSub(DataType t, wasm::Expression* a, wasm::Expression* b);
  wasm::Expression* CreateMul(DataType t, wasm::Expression* a, wasm::Expression* b);

  wasm::Expression* CreateSerialFor(wasm::Expression* begin,
                                    wasm::Expression* end,
                                    wasm::Expression* stride,
                                    const Var& loop_var,
                                    const Stmt& body);

  wasm::Expression* CreateCallExtern(Type ret_type,
                                     String global_symbol,
                                     const Array<PrimExpr>& args,
                                     bool skip_first_arg);

  void WasmTypeToDLDataType(wasm::Type type, int32_t* dtype_code_ptr, int32_t* dtype_bits_ptr);

  void AddWasmExportedMemory();

  // SeqStmt debug.
  int seq_stmt_depth_;

  wasm::Type t_void_{wasm::Type::BasicType::i32};
  wasm::Type t_void_p_{wasm::Type::BasicType::i32};
  int native_vector_bits_{128};
  std::unordered_map<const VarNode*, StorageInfo> alloc_storage_info_;
  wasm::Name default_memory_name_{"0"};
  wasm::Address default_initial_memory_size_{16 * 1024};
  wasm::Address default_max_memory_size_{int32_t(64 * 1024)};
  std::unique_ptr<arith::Analyzer> analyzer_;
  std::unique_ptr<wasm::Function> function_;
  std::unique_ptr<wasm::Module> module_;
  std::unique_ptr<wasm::Builder> builder_;
  wasm::NameType last_wasm_var_;
  wasm::Type last_wasm_var_dtype_;
  std::vector<wasm::NameType> wasm_params_;
  std::vector<wasm::NameType> wasm_vars_;
  std::unordered_map<const VarNode*, wasm::NameType> var_map_;
  // Buffer to vars.
  bool enable_buf_to_var_{false};
  std::unordered_map<const wasm::NameType*, const std::vector<wasm::NameType>*> buf_cache_var_map_;
  std::unordered_map<std::pair<std::pair<const VarNode*, const VarNode*>, int64_t>, wasm::NameType> load_cache_var_map_;
  // Cached cse vars.
  std::unordered_map<std::pair<const VarNode*, const VarNode*>, int64_t> cse_var_offset_map_;
  std::vector<const VarNode*> cached_cse_vars_;
  // Body expressions;
  std::vector<std::vector<wasm::Expression*>> stmt_body_exprs_;

  OpAttrMap<TGlobalSymbol> op_attr_global_symbol_ = Op::GetAttrMap<TGlobalSymbol>("TGlobalSymbol");
  const Op& builtin_call_extern_ = builtin::call_extern();
  const Op& builtin_call_pure_extern_ = builtin::call_pure_extern();
};

template <typename IterType, typename ConvType>
void CodeGenWasm::AddFunctionsOrdered(IterType begin, IterType end, ConvType pfunc) {
  std::vector<std::tuple<GlobalVar, PrimFunc>> funcs;
  for (auto it = begin; it != end; ++it) {
    auto [gvar, func] = *it;
    auto converted = pfunc(func);
    funcs.push_back({gvar, Downcast<PrimFunc>(converted)});
  }
  std::sort(funcs.begin(), funcs.end(), [this](const auto& pair_a, const auto& pair_b) {
    const auto& [gvar_a, func_a] = pair_a;
    std::string name_a = GetSymbolName(gvar_a, func_a);

    const auto& [gvar_b, func_b] = pair_b;
    std::string name_b = GetSymbolName(gvar_b, func_b);
    return name_a < name_b;
  });

  for (const auto& [gvar, func] : funcs) {
    AddFunction(func);
  }
}

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_TARGET_WASM_CODEGEN_WASM_H_
