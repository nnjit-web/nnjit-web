/**
 * TVM JS Wasm Runtime library.
 */
import { Pointer, PtrOffset } from "./ctypes";
import { Disposable } from "./types";
import { Memory, CachedCallStack } from "./memory";
import { Environment } from "./environment";
import { WebGPUContext } from "./webgpu";
/**
 * Type for PackedFunc inthe TVMRuntime.
 */
export type PackedFunc = ((...args: any) => any) & Disposable & {
    _tvmPackedCell: PackedFuncCell;
};
/**
 * @internal
 * FFI Library wrapper, maintains most runtime states.
 */
declare class FFILibrary implements Disposable {
    wasm32: boolean;
    memory: Memory;
    exports: Record<string, Function>;
    webGPUContext?: WebGPUContext;
    private wasmInstance;
    private recycledCallStacks;
    constructor(wasmInstance: WebAssembly.Instance, imports: Record<string, any>);
    dispose(): void;
    sizeofPtr(): number;
    checkCall(code: number): void;
    getOrAllocCallStack(): CachedCallStack;
    recycleCallStack(callstack: CachedCallStack): void;
    private validateInstance;
    private checkExports;
    private detectWasmMemory;
}
/**
 * A typed scalar constant used to represent a typed number
 * argument to PackedFunc calls.
 */
export declare class Scalar {
    /** The value. */
    value: number;
    /** The data type of the scalar. */
    dtype: string;
    constructor(value: number, dtype: string);
}
/**
 * Cell holds the PackedFunc object.
 */
declare class PackedFuncCell implements Disposable {
    handle: Pointer;
    private lib;
    constructor(handle: Pointer, lib: FFILibrary);
    dispose(): void;
}
/**
 * Represent a runtime context where a NDArray can reside.
 */
export declare class DLDevice {
    /** The device type code of the device. */
    deviceType: number;
    /** The device index. */
    deviceId: number;
    private lib;
    constructor(deviceType: number | string, deviceId: number, lib: FFILibrary);
    /**
     * Synchronize the device
     */
    sync(): Promise<void>;
    toString(): string;
}
/**
 * The data type code in DLDataType
 */
export declare const enum DLDataTypeCode {
    Int = 0,
    UInt = 1,
    Float = 2,
    OpaqueHandle = 3
}
/**
 * Runtime data type of NDArray.
 */
export declare class DLDataType {
    /** The type code */
    code: number;
    /** Number of bits in the data type. */
    bits: number;
    /** Number of vector lanes. */
    lanes: number;
    constructor(code: number, bits: number, lanes: number);
    toString(): string;
    numStorageBytes(): number;
}
/**
 * n-dimnesional array.
 */
export declare class NDArray implements Disposable {
    /** Internal array handle. */
    handle: Pointer;
    /** Number of dimensions. */
    ndim: number;
    /** Data type of the array. */
    dtype: string;
    /** Shape of the array. */
    shape: Array<number>;
    /** Device of the array. */
    device: DLDevice;
    /** Whether it is a temporary view that can become invalid after the call. */
    private isView;
    private byteOffset;
    private dltensor;
    private dataPtr;
    private lib;
    private dlDataType;
    constructor(handle: Pointer, isView: boolean, lib: FFILibrary);
    dispose(): void;
    /**
     * Copy data from another NDArray or javascript array.
     * The number of elements must match.
     *
     * @param data The source data array.
     * @returns this
     */
    copyFrom(data: NDArray | Array<number> | Float32Array): this;
    /**
     * Copy data from raw bytes.
     * @param data Uint8Array of bytes.
     * @returns this
     */
    copyFromRawBytes(data: Uint8Array): this;
    /**
     * Return a copied Uint8Array of the raw bytes in the NDArray.
     * @returns The result array.
     */
    toRawBytes(): Uint8Array;
    /**
     * Return a TypedArray copy of the NDArray, the specific type depends on
     * the dtype of the NDArray.
     * @returns The result array.
     */
    toArray(): Float32Array | Float64Array | Int32Array | Int8Array | Uint8Array;
    private getDLTensorFromArrayHandle;
}
/**
 * Runtime Module.
 */
export declare class Module implements Disposable {
    handle: Pointer;
    private lib;
    private makePackedFunc;
    constructor(handle: Pointer, lib: FFILibrary, makePackedFunc: (ptr: Pointer) => PackedFunc);
    dispose(): void;
    /**
     * Get a function in the module.
     * @param name The name of the function.
     * @returns The result function.
     */
    getFunction(name: string): PackedFunc;
    /**
     * Import another module into the current runtime module.
     * @param mod The module to be imported.
     */
    importModule(mod: Module): void;
}
/**
 *  Graph executor.
 *
 *  This is a thin wrapper of the underlying TVM module.
 *  you can also directly call set_input, run, and get_output
 *  of underlying module functions
 */
declare class GraphExecutor implements Disposable {
    module: Module;
    private packedSetInput;
    private packedRun;
    private packedGetOutput;
    private packedLoadParams;
    /**
     * COnstructor
     * @param module The underlying module.
     */
    constructor(module: Module);
    dispose(): void;
    /**
     * Set input to the executor.
     *
     * @param key The input key.
     * @param value The value to get set.
     */
    setInput(key: number | string, value: NDArray): void;
    /**
     * Execute the underlying graph.
     */
    run(): void;
    /**
     * Get index-th output.
     * @param index The index number.
     * @param out The optional output storage parameters.
     * @returns The output array.
     */
    getOutput(index: number, out?: NDArray | undefined): NDArray;
    /**
     * Load parameters from parameter binary.
     * @param paramBinary The parameter binary.
     */
    loadParams(paramBinary: Uint8Array): void;
    /**
     * Benchmark stable execution of the graph(without data copy).
     * @params dev The device to sync during each run.
     * @number The number of times to compute the average.
     * @repeat The number of times to repeat the run.
     */
    benchmarkRuns(dev: DLDevice, number?: number, repeat?: number): Promise<number[]>;
}
declare class GraphExecutorDebug extends GraphExecutor {
    private packedRunIndividual;
    private packedProfileRpc;
    constructor(module: Module);
    runIndividual(number?: number, repeat?: number, minRepeatMs?: number, limitZeroTimeIterations?: number, cooldownIntervalMs?: number, repeatsToCooldown?: number): Uint8Array;
    profileRpc(): string;
}
/**
 * TVM runtime instance.
 */
export declare class Instance implements Disposable {
    memory: Memory;
    exports: Record<string, Function>;
    private lib;
    private env;
    private isTimeEvalFinished;
    private timeEvalResults;
    private graphExecutor;
    private graphExecutorDebug;
    /**
     * Internal function(registered by the runtime)
     */
    private wasmCreateLibraryModule?;
    /**
     * Constructor
     *
     * importObject can also be a {@link LibraryProvider} object,
     * a WASI object, or an object containing wasmLibraryProvider field.
     *
     * @param wasmModule The input module or instance.
     * @param importObject The imports to initialize the wasmInstance if it is not provided.
     * @param wasmInstance Additional wasm instance argument for deferred construction.
     * @param env Directly specified environment module.
     *
     * @see Please use the async version {@link instantiate} when targeting browsers.
     */
    constructor(wasmModule: WebAssembly.Module, importObject?: Record<string, any>, wasmInstance?: WebAssembly.Instance, env?: Environment);
    dispose(): void;
    /**
     * Get system-wide library module in the wasm.
     * System lib is a global module that contains self register functions in startup.
     * @returns The system library module.
     */
    systemLib(): Module;
    /**
     * List all the global function names registered in the runtime.
     * @returns The name list.
     */
    listGlobalFuncNames(): Array<string>;
    /**
     * Register function to be global function in tvm runtime.
     * @param name The name of the function.
     * @param f function to be registered.
     * @param override Whether overwrite function in existing registry.
     */
    registerFunc(name: string, func: PackedFunc | Function, override?: boolean): void;
    /**
     * Get global PackedFunc from the runtime.
     * @param name The name of the function.
     * @returns The result function.
     */
    getGlobalFunc(name: string): PackedFunc;
    /**
     * Check if func is PackedFunc.
     *
     * @param func The input.
     * @returns The check result.
     */
    isPackedFunc(func: unknown): boolean;
    /**
     * Convert func to PackedFunc
     *
     * @param func Input function.
     * @returns The converted function.
     */
    toPackedFunc(func: Function): PackedFunc;
    /**
     * Convert dtype to {@link DLDataType}
     *
     * @param dtype The input dtype string or DLDataType.
     * @returns The converted result.
     */
    toDLDataType(dtype: string | DLDataType): DLDataType;
    /**
     * Create a new {@link Scalar} that can be passed to a PackedFunc.
     * @param value The number value.
     * @param dtype The dtype string.
     * @returns The created scalar.
     */
    scalar(value: number, dtype: string): Scalar;
    /**
     * Create a new {@link DLDevice}
     * @param deviceType The device type.
     * @param deviceId The device index.
     * @returns The created device.
     */
    device(deviceType: number | string, deviceId?: number): DLDevice;
    /**
     * Create a new cpu {@link DLDevice}
     * @param deviceId The device index.
     */
    cpu(deviceId?: number): DLDevice;
    /**
     * Create a new webgpu {@link DLDevice}
     * @param deviceId The device index.
     */
    webgpu(deviceId?: number): DLDevice;
    /**
     * Create an empty {@link NDArray} with given shape and dtype.
     *
     * @param shape The shape of the array.
     * @param dtype The data type of the array.
     * @param dev The device of the ndarray.
     * @returns The created ndarray.
     */
    empty(shape: Array<number> | number, dtype?: string | DLDataType, dev?: DLDevice): NDArray;
    /**
     * Create a new graph executor.
     *
     * @param graphJson The graph executor json file.
     * @param lib The underlying library.
     * @param dev The execution device of the graph.
     */
    createGraphExecutor(graphJson: string, lib: Module, dev: DLDevice): GraphExecutor;
    createGraphExecutorDebug(graphJson: string, lib: Module, dev: DLDevice): GraphExecutorDebug;
    /**
     * Register an asyncfunction to be global function in the server.
     * @param name The name of the function.
     * @param func function to be registered.
     * @param override Whether overwrite function in existing registry.
     *
     * @note The async function will only be used for serving remote calls in the rpc.
     */
    registerAsyncServerFunc(name: string, func: Function, override?: boolean): void;
    registerSyncServerFunc(name: string, func: Function, override?: boolean): void;
    /**
     * Initialize webgpu in the runtime.
     * @param device The given GPU device.
     */
    initWebGPU(device: GPUDevice): void;
    /** Register global packed functions needed by the backend to the env. */
    private registerEnvGlobalPackedFuncs;
    private createPackedFuncFromCFunc;
    /**
     * Set packed function arguments into the location indicated by argsValue and argsCode.
     * Allocate new temporary space from the stack if necessary.
     *
     * @parma stack The call stack
     * @param args  The input arguments.
     * @param argsValue The offset of argsValue.
     * @param argsCode The offset of argsCode.
     */
    setPackedArguments(stack: CachedCallStack, args: Array<any>, argsValue: PtrOffset, argsCode: PtrOffset): void;
    private wrapJSFuncAsPackedCFunc;
    private makePackedFunc;
    private retValueToJS;
}
/**
 * Asynchrously instantiate a new {@link Instance}.
 *
 * importObject can also be a {@link LibraryProvider} object,
 * a WASI object, or an object containing wasmLibraryProvider field.
 * We can take benefit of syslib implementations from the Emscripten
 * by passing its generated js Module as the imports.
 *
 * @param bufferSource The source to be compiled.
 * @param importObject The import objects.
 * @param logger The system logger.
 */
export declare function instantiate(bufferSource: ArrayBuffer, importObject?: Record<string, any>, logger?: (msg: string) => void): Promise<Instance>;
export {};
