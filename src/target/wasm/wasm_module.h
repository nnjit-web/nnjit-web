
#ifndef TVM_TARGET_WASM_WASM_MODULE_H_
#define TVM_TARGET_WASM_WASM_MODULE_H_


#include <tvm/relay/runtime.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/metadata.h>
#include <tvm/runtime/module.h>
#include <tvm/target/target.h>

namespace tvm {
namespace codegen {

runtime::Module CreateWasmMetadataModule(
    const Array<runtime::Module>& modules, Target target,
    tvm::relay::Runtime runtime);

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_TARGET_WASM_WASM_MODULE_H_
