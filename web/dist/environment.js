"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.Environment = void 0;
const support_1 = require("./support");
/**
 * Detect library provider from the importObject.
 *
 * @param importObject The import object.
 */
function detectLibraryProvider(importObject) {
    if (importObject["wasmLibraryProvider"] &&
        importObject["wasmLibraryProvider"]["start"] &&
        importObject["wasmLibraryProvider"]["imports"] !== undefined) {
        const item = importObject;
        // create provider so that we capture imports in the provider.
        return {
            imports: item.wasmLibraryProvider.imports,
            start: (inst) => {
                item.wasmLibraryProvider.start(inst);
            },
        };
    }
    else if (importObject["imports"] && importObject["start"] !== undefined) {
        return importObject;
    }
    else if (importObject["wasiImport"] && importObject["start"] !== undefined) {
        // WASI
        return {
            imports: {
                "wasi_snapshot_preview1": importObject["wasiImport"],
            },
            start: (inst) => {
                //importObject["start"](inst);
                importObject["initialize"](inst);
            }
        };
    }
    else {
        return undefined;
    }
}
/**
 * Environment to impelement most of the JS library functions.
 */
class Environment {
    constructor(importObject = {}, logger = console.log) {
        /**
         * Maintains a table of FTVMWasmPackedCFunc that the C part
         * can call via TVMWasmPackedCFunc.
         *
         * We maintain a separate table so that we can have un-limited amount
         * of functions that do not maps to the address space.
         */
        this.packedCFuncTable = [
            undefined,
        ];
        /**
         * Free table index that can be recycled.
         */
        this.packedCFuncTableFreeId = [];
        this.logger = logger;
        this.libProvider = detectLibraryProvider(importObject);
        // get imports from the provider
        if (this.libProvider !== undefined) {
            this.imports = this.libProvider.imports;
        }
        else {
            this.imports = importObject;
        }
        // update with more functions
        this.imports.env = this.environment(this.imports.env);
    }
    /** Mark the start of the instance. */
    start(inst) {
        if (this.libProvider !== undefined) {
            this.libProvider.start(inst);
        }
    }
    environment(initEnv) {
        // default env can be be overriden by libraries.
        const defaultEnv = {
            "__cxa_thread_atexit": () => { },
            // eslint-disable-next-line @typescript-eslint/no-unused-vars
            "emscripten_notify_memory_growth": (index) => { }
        };
        const wasmPackedCFunc = (args, typeCodes, nargs, ret, resourceHandle) => {
            const cfunc = this.packedCFuncTable[resourceHandle];
            (0, support_1.assert)(cfunc !== undefined);
            return cfunc(args, typeCodes, nargs, ret, resourceHandle);
        };
        const wasmPackedCFuncFinalizer = (resourceHandle) => {
            this.packedCFuncTable[resourceHandle] = undefined;
            this.packedCFuncTableFreeId.push(resourceHandle);
        };
        const newEnv = {
            TVMWasmPackedCFunc: wasmPackedCFunc,
            TVMWasmPackedCFuncFinalizer: wasmPackedCFuncFinalizer,
            "__console_log": (msg) => {
                this.logger(msg);
            }
        };
        return Object.assign(defaultEnv, initEnv, newEnv);
    }
}
exports.Environment = Environment;
//# sourceMappingURL=environment.js.map