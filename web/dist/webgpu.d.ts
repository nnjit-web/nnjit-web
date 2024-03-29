import "@webgpu/types";
import { Memory } from "./memory";
/** A pointer to points to the raw address space. */
export type GPUPointer = number;
/**
 * DetectGPU device in the environment.
 */
export declare function detectGPUDevice(): Promise<GPUDevice | undefined | null>;
/**
 * WebGPU context
 * Manages all the webgpu resources here.
 */
export declare class WebGPUContext {
    device: GPUDevice;
    memory: Memory;
    pipeline: GPUComputePipeline | undefined;
    timeQuerySet: any;
    logger: (msg: string) => void;
    private bufferTable;
    private bufferTableFreeId;
    private bufferSizeTable;
    private pendingRead;
    private numPendingReads;
    private shaderNames;
    private shaderFuncMap;
    private isProfilingSupportedFlag;
    private enableProfilingFlag;
    private numQueryCount;
    private curTimeQueryIdx;
    private timestampBufferSize;
    private timestampBufferIdx;
    private timestampOutBufferIdx;
    private compilationStartTime;
    private compilationStatus;
    private compilationInfo;
    constructor(memory: Memory, device: GPUDevice);
    isProfilingSupported(): boolean;
    enableProfiling(): void;
    disbaleProfiling(): void;
    isProfilingEnabled(): boolean;
    initTimeQuery(): void;
    resetTimeQuery(): void;
    resolveQuerySet(): Promise<void>;
    getTimeCostArray(unit: String, outArr: Array<number>): Promise<void>;
    private log;
    getShaderNames(): Array<string>;
    /**
     * Wait for all pending GPU tasks to complete
     */
    /**
    async sync(): Promise<void> {
      const fence = this.device.queue.createFence();
      this.device.queue.signal(fence, 1);
      if (this.numPendingReads != 0) {
        // eslint-disable-next-line @typescript-eslint/no-empty-function
        await Promise.all([fence.onCompletion(1), this.pendingRead]);
      } else {
        await fence.onCompletion(1);
      }
    }
    */
    sync(): Promise<undefined>;
    /**
     * Create a PackedFunc that runs the given shader
     *
     * @param info The function information in json.
     * @param data The shader data(in SPIRV)
     */
    createShader(name: string, info: string, data: Uint8Array): Function;
    executeAfterCompilation(callbackForSuccess: Function, callbackForFail: Function): void;
    /**
     * Get the device API according to its name
     * @param The name of the API.
     * @returns The corresponding device api.
     */
    getDeviceAPI(name: string): Function;
    private deviceAllocDataSpace;
    private deviceAllocDataSpaceForQueryResolve;
    private deviceAllocDataSpaceForQueryResolveResult;
    private deviceFreeDataSpace;
    private deviceCopyToGPU;
    private deviceCopyFromGPU;
    private deviceCopyWithinGPU;
    private gpuBufferFromPtr;
    private attachToBufferTable;
}
