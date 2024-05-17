
#include "wasm_module.h"

#include <wasm-builder.h>
#include <wasm-io.h>

#include <tvm/ir/module.h>
#include <tvm/relay/runtime.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/metadata.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/support/with.h>
#include <tvm/target/codegen.h>
#include <tvm/target/target.h>

#include "../../runtime/file_utils.h"
#include "../../runtime/library_module.h"
#include "../func_registry_generator.h"
#include "codegen_wasm.h"

namespace tvm {
namespace codegen {

class WasmModuleNode final : public runtime::ModuleNode {
 public:
  ~WasmModuleNode();

  const char* type_key() const final { return "wasm"; }

  PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) final;

  int GetPropertyMask() const override {
    return runtime::ModulePropertyMask::kRunnable | runtime::ModulePropertyMask::kDSOExportable;
  }

  void SaveToFile(const String& file_name, const String& format) final;
  void SaveToBinary(dmlc::Stream* stream) final;
  String GetSource(const String& format) final;

  void Init(const IRModule& mod, const Target& target);
  void Init(std::unique_ptr<wasm::Module> module);

  bool ImplementsFunction(const String& name, bool query_imports) final;

 private:
  wasm::Module* module_{nullptr};
  std::unique_ptr<wasm::Module> module_owning_ptr_;
  Array<String> function_names_;
};

WasmModuleNode::~WasmModuleNode() {
  module_owning_ptr_.reset();
}

PackedFunc WasmModuleNode::GetFunction(const String& name,
                                       const ObjectPtr<Object>& sptr_to_self) {
  LOG(FATAL) << "WasmModule: GetFunction not supported";
  return PackedFunc();
}

void WasmModuleNode::SaveToFile(const String& file_name, const String& format) {
  std::string fmt = runtime::GetFileFormat(file_name, format);
  wasm::PassOptions pass_options;
  wasm::ModuleWriter writer(pass_options);
  if (fmt == "wasm") {
    writer.setBinary(true);
  } else if (fmt == "wat") {
    writer.setBinary(false);
  } else {
    LOG(FATAL) << "Do not know how to save file " << file_name << " with format=\'" << format
               << "\'";
  }
  writer.setDebugInfo(true);
  writer.write(*module_, file_name);
}

void WasmModuleNode::SaveToBinary(dmlc::Stream* stream) {
  LOG(FATAL) << "WasmModule: SaveToBinary not supported";
}

String WasmModuleNode::GetSource(const String& format) {
  LOG(FATAL) << "WasmModule: GetSource not supported";
  return "";
}

void WasmModuleNode::Init(const IRModule& mod, const Target& target) {
  std::unique_ptr<CodeGenWasm> cg = std::make_unique<CodeGenWasm>();

  std::vector<PrimFunc> funcs;
  std::string entry_func;
  for (auto kv : mod->functions) {
    if (!kv.second->IsInstance<PrimFuncNode>()) {
      DLOG(INFO) << "Can only lower IR Module with PrimFuncs, but got " << kv.second->GetTypeKey();
      continue;
    }
    auto f = Downcast<PrimFunc>(kv.second);
    auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
    ICHECK(global_symbol.defined());
    function_names_.push_back(global_symbol.value());
    if (f->HasNonzeroAttr(tir::attr::kIsEntryFunc)) {
      entry_func = global_symbol.value();
    }
    funcs.push_back(f);
  }
  
  try {
    cg->Init();
    cg->AddFunctionsOrdered(funcs.begin(), funcs.end());
    if (entry_func.length() != 0) {
      cg->AddMainFunction(entry_func);
    }
  } catch (const Error& e) {
    DLOG(INFO) << "Add function error: " << e.what();
  }

  module_owning_ptr_ = cg->Finish();
  module_ = module_owning_ptr_.get();
}

void WasmModuleNode::Init(std::unique_ptr<wasm::Module> module) {
  module_owning_ptr_ = std::move(module);
  module_ = module_owning_ptr_.get();
}

bool WasmModuleNode::ImplementsFunction(const String& name, bool query_imports) {
  LOG(FATAL) << "WasmModule: ImplementsFunction not supported";
  return false;
}

TVM_REGISTER_GLOBAL("target.build.wasm")
    .set_body_typed([](IRModule mod, Target target) -> runtime::Module {
      auto n = make_object<WasmModuleNode>();
      n->Init(mod, target);
      return runtime::Module(n);
    });

TVM_REGISTER_GLOBAL("codegen.wasm_target_enabled")
    .set_body_typed([](std::string target_str) -> bool {
      return true;
    });

runtime::Module CreateWasmMetadataModule(
    const Array<runtime::Module>& modules,
    Target target,
    tvm::relay::Runtime runtime) {
  auto cg = std::make_unique<CodeGenWasm>();
  cg->Init();
  auto mod = cg->Finish();

  auto n = make_object<WasmModuleNode>();
  n->Init(std::move(mod));
  for (auto m : modules) {
    n->Import(m);
  }
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.CreateWasmMetadataModule")
    .set_body_typed(CreateWasmMetadataModule);

}  // namespace codegen
}  // namespace tvm
