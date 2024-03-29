/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
import "@webgpu/types";
import { assert } from "./support";
import { Pointer } from "./ctypes";
import { Memory } from "./memory";
import * as compact from "./compact";

/** A pointer to points to the raw address space. */
export type GPUPointer = number;

/**
 * DetectGPU device in the environment.
 */
export async function detectGPUDevice(): Promise<GPUDevice | undefined | null> {
  if (typeof navigator !== "undefined" && navigator.gpu !== undefined) {
    const adapter = await navigator.gpu.requestAdapter();
    const supportTimeQuery = adapter?.features.has("timestamp-query");
    const deviceDescriptor: GPUDeviceDescriptor = {};
    if (supportTimeQuery) {
      deviceDescriptor.requiredFeatures = ["timestamp-query"];
    }
    return await adapter?.requestDevice(deviceDescriptor);
  } else {
    return undefined;
  }
}

interface FunctionInfo {
  name: string;
  arg_types: Array<string>;
  launch_param_tags: Array<string>;
}

enum ShaderModuleCompilationStatus {
  "none",
  "compiling",
  "failed",
  "successful"
}

/**
 * WebGPU context
 * Manages all the webgpu resources here.
 */
export class WebGPUContext {
  device: GPUDevice;
  memory: Memory;
  pipeline: GPUComputePipeline | undefined;
  timeQuerySet: any;
  logger: (msg: string) => void;

  //private readBuffer:;
  private bufferTable: Array<GPUBuffer | undefined> = [undefined];
  private bufferTableFreeId: Array<number> = [];
  private bufferSizeTable: Array<number> = [];
  private pendingRead: Promise<void> = Promise.resolve();
  private numPendingReads = 0;
  private shaderNames: Array<string> = [];
  private shaderFuncMap: Map<string, Function> = new Map();

  private isProfilingSupportedFlag = false;
  private enableProfilingFlag = false;
  private numQueryCount = 128;
  private curTimeQueryIdx: GPUSize32 = 0;
  private timestampBufferSize: number = -1;
  private timestampBufferIdx: number = -1;
  private timestampOutBufferIdx: number = -1;

  private compilationStartTime: number = 0;
  private compilationStatus: ShaderModuleCompilationStatus = ShaderModuleCompilationStatus.none;
  private compilationInfo: string = "";

  constructor(memory: Memory, device: GPUDevice) {
    this.memory = memory;
    this.device = device;
    this.pipeline = undefined;
    try {
      this.timeQuerySet = this.device.createQuerySet({
          type: "timestamp", count: this.numQueryCount});
      this.isProfilingSupportedFlag = true;
    } catch (err) {
      this.timeQuerySet = undefined;
      this.isProfilingSupportedFlag = false;
    }
    this.logger = console.log;
    this.log("supportTimestampQuerySet " + this.isProfilingSupportedFlag);
  }

  isProfilingSupported() {
    return this.isProfilingSupportedFlag;
  }

  enableProfiling() {
    if (this.isProfilingSupportedFlag == true) {
      this.enableProfilingFlag = true;
      this.initTimeQuery();
    }
  }

  disbaleProfiling() {
    if (this.isProfilingSupportedFlag == true) {
      this.enableProfilingFlag = false;
    }
  }

  isProfilingEnabled() {
    return this.isProfilingSupportedFlag && this.enableProfilingFlag;
  }

  initTimeQuery() {
    if (this.isProfilingEnabled() == false) {
      return;
    }
    this.curTimeQueryIdx = 0;
    this.timestampBufferSize = this.numQueryCount * 8 + 8 * 8;
    if (this.timestampBufferIdx < 0) {
      this.timestampBufferIdx = this.deviceAllocDataSpaceForQueryResolve(this.timestampBufferSize);
      this.timestampOutBufferIdx = this.deviceAllocDataSpaceForQueryResolveResult(this.timestampBufferSize);
    }
  }

  resetTimeQuery() {
    if (this.isProfilingEnabled() == false) {
      return;
    }
    this.curTimeQueryIdx = 0;
  }

  async resolveQuerySet() {
    const timeQueryCount = this.curTimeQueryIdx;
    const commandEncoder = this.device.createCommandEncoder();
    for (let i = 0; i < timeQueryCount; i += 32) {
      commandEncoder.resolveQuerySet(
          this.timeQuerySet,
          i,
          32,
          this.gpuBufferFromPtr(this.timestampBufferIdx),
          (i / 32) * 256);
    }
    const command = commandEncoder.finish();
    this.device.queue.submit([command]);
    this.deviceCopyWithinGPU(
        this.timestampBufferIdx, 0,
        this.timestampOutBufferIdx, 0,
        this.timestampBufferSize);
    await this.sync();
  }

  async getTimeCostArray(unit: String, outArr: Array<number>)  {
    if (this.isProfilingEnabled() == false) {
      return;
    }
    if (this.timestampBufferIdx >= 0) {
      await this.resolveQuerySet();
      const timestampBuffer = this.gpuBufferFromPtr(this.timestampOutBufferIdx);
      await timestampBuffer.mapAsync(GPUMapMode.READ);
      const cpuTemp = timestampBuffer.getMappedRange();
      const viewU64 = new BigUint64Array(cpuTemp);

      for (let i = 0; i < this.curTimeQueryIdx; i += 2) {
        const durationNs = viewU64[i + 1] - viewU64[i];
        let speed = Number(durationNs);
        if (unit == "s") {
          speed = speed / 1e9;
        } else if (unit == "ms") {
          speed = speed / 1e6;
        } else if (unit == "us") {
          speed = speed / 1e3;
        }

        outArr.push(speed);
      }

      timestampBuffer.unmap();
    }
  }

  private log(msg: string): void {
    this.logger("WebGPUContext: " + msg);
  }

  getShaderNames(): Array<string> {
    return this.shaderNames;
  }

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

  async sync(): Promise<undefined> {
    return this.device.queue.onSubmittedWorkDone();
  }

  /**
   * Create a PackedFunc that runs the given shader
   *
   * @param info The function information in json.
   * @param data The shader data(in SPIRV)
   */
  createShader(name: string, info: string, data: Uint8Array): Function {
    if (!this.shaderNames.includes(name)) {
      this.shaderNames.push(name);
    }
    if (this.shaderFuncMap.get(name) != undefined) {
      return this.shaderFuncMap.get(name)!;
    }
    //if (!name.includes("default_function_kernel") && !name.includes("batch_matmul")) {
    //  const submitShader = (...args: Array<GPUPointer | number>): void => {};
    //  this.shaderFuncMap.set(name, submitShader);
    //  return submitShader;
    //}

    const finfo = JSON.parse(info);
    const layoutEntries: Array<GPUBindGroupLayoutEntry> = [];
    for (let i = 0; i < finfo.arg_types.length; ++i) {
      const dtype = finfo.arg_types[i];
      if (dtype == "handle") {
        layoutEntries.push({
          binding: i,
          visibility: GPUShaderStage.COMPUTE,
          //type: "storage-buffer",
          buffer: {
            type: "storage"
          }
        });
      } else {
        throw new Error("Cannot handle argument type " + dtype + " in WebGPU shader");
      }
    }
    const bindGroupLayout = this.device.createBindGroupLayout({
      entries: layoutEntries
    });

    const getDateString = (): string => {
      const date = new Date();
      return date.getFullYear() + "/" + (date.getMonth() + 1) + "/" + date.getDate()
          + " " + date.getHours() + ":" + date.getMinutes() + ":" + date.getSeconds();
    };

    const srcCodeBytes = data.buffer.byteLength;
    const dateStartStr = getDateString();
    this.compilationStatus = ShaderModuleCompilationStatus.compiling;
    this.compilationInfo = "bytes " + srcCodeBytes + ", start " + dateStartStr;
    const maxSrcCodeBytes = 512 * 1024;

    if (srcCodeBytes > maxSrcCodeBytes) {
      this.compilationStatus = ShaderModuleCompilationStatus.failed;
      const submitNothing = (...args: Array<GPUPointer | number>): void => {
      };
      return submitNothing;
    }
    
    const srcCode = new Uint32Array(data.buffer);
    //const srcCode = String.fromCharCode.apply(null, data);
    //const srcCode = new TextDecoder().decode(data);

    const perf = compact.getPerformance();
    let tstart: number = perf.now();
    const shaderModule = this.device.createShaderModule({
      code: srcCode
    });
    let tend: number = perf.now();
    let durationMs: number = tend - tstart;
    this.log("createShaderModule " + durationMs + " ms");
    
    this.device.pushErrorScope("validation");
    const timeout = 60 * 1000;  // ms
    this.compilationStartTime = tstart = perf.now();
    let pipeline: GPUComputePipeline | undefined = undefined;
    let enableAsync = true;
    if (!name.includes("default_function_kernel")) {
      enableAsync = false;
    }
    if (!enableAsync) {
      pipeline = this.device.createComputePipeline({
        layout: this.device.createPipelineLayout({
          bindGroupLayouts: [ bindGroupLayout ]
        }),
        compute: {
          module: shaderModule,
          entryPoint: "main"
        }
      });
      tend = perf.now();
      durationMs = tend - tstart;
      this.log("createComputePipeline: name " + name + ", bytes " + srcCodeBytes
          + ", start " + dateStartStr + ", dura " + durationMs + " ms");

      this.device.popErrorScope().then((error) => {
        if (error) {
          this.compilationStatus = ShaderModuleCompilationStatus.failed;
          this.log("createComputePipeline: error, name " + name);
          if (error instanceof GPUValidationError) {
            this.log("Error: " + error.message);
          }
        } else {
          this.compilationStatus = ShaderModuleCompilationStatus.successful;
        }
      });
    } else {
      this.device.createComputePipelineAsync({
        layout: this.device.createPipelineLayout({
          bindGroupLayouts: [ bindGroupLayout ]
        }),
        compute: {
          module: shaderModule,
          entryPoint: "main"
        }
      }).then((outputPipeline) => {
        tend = perf.now();
        durationMs = tend - tstart;
        this.log("createComputePipeline: bytes " + srcCodeBytes
            + ", start " + dateStartStr + ", dura " + durationMs + " ms");

        if (durationMs > timeout) {
          this.compilationStatus = ShaderModuleCompilationStatus.failed;
          this.log("createComputePipeline: timeout");
        } else {
          pipeline = outputPipeline;
          this.device.popErrorScope().then((error) => {
            if (error) {
              this.compilationStatus = ShaderModuleCompilationStatus.failed;
              this.log("createComputePipeline: error");
              if (error instanceof GPUValidationError) {
                this.log("Error: " + error.message);
              }
            } else {
              this.compilationStatus = ShaderModuleCompilationStatus.successful;
            }
          });
        }
      }).catch((error: DOMException) => {
        tend = perf.now();
        durationMs = tend - tstart;
        this.log("createComputePipeline: bytes " + srcCodeBytes
            + ", start " + dateStartStr + ", dura " + durationMs + " ms");

        this.compilationStatus = ShaderModuleCompilationStatus.failed;
        this.log("createComputePipeline: error");
        if (error instanceof GPUValidationError) {
          this.log("Error: " + error.message);
        }
      });
    }

    //tstart = perf.now();
    const dispatchToDim: Array<number> = [];

    for (let i = 0; i < finfo.launch_param_tags.length; ++i) {
      const tag: string = finfo.launch_param_tags[i];
      if (tag.startsWith("blockIdx.")) {
        const target: number = tag.charCodeAt(tag.length - 1) - ("x".charCodeAt(0));
        assert(target >= 0 && target < 3);
        dispatchToDim.push(target);
      } else if (tag.startsWith("threadIdx.")) {
        const target: number = tag.charCodeAt(tag.length - 1) - ("x".charCodeAt(0));
        assert(target >= 0 && target < 3);
        dispatchToDim.push(target + 3);
      } else {
        throw new Error("Cannot handle thread_axis " + tag);
      }
    }
    //tend = perf.now();
    //durationMs = tend - tstart;
    //this.log("setDispatchToDim " + durationMs + " ms");

    const submitShader = (...args: Array<GPUPointer | number>): void => {
      // Limit kernel name.
      //if (!name.includes("default_function_kernel") && !name.includes("batch_matmul")) {
      //  return;
      //}
      //if (name.includes("_mean_")) {
      //  return;
      //}
      
      // Check buffer size.
      for (let i = 0; i < layoutEntries.length; ++i) {
        const bufferSize = this.bufferSizeTable[args[i]];
        if (bufferSize == 0 || bufferSize > 128 * 1024 * 1024) {
          return;
        }
      }
      
      this.log("submitShader: name " + name);
      if (pipeline == undefined) {
        return;
      }
      const commandEncoder = this.device.createCommandEncoder();
      if (this.isProfilingEnabled()) {
        if (this.curTimeQueryIdx < this.numQueryCount) {
          commandEncoder.writeTimestamp(this.timeQuerySet, this.curTimeQueryIdx);
        }
      }
      const compute = commandEncoder.beginComputePass();
      compute.setPipeline(pipeline);
      const bindGroupEntries: Array<GPUBindGroupEntry> = [];
      assert(args.length == layoutEntries.length + dispatchToDim.length);

      for (let i = 0; i < layoutEntries.length; ++i) {
        bindGroupEntries.push({
          binding: i,
          resource: {
            buffer: this.gpuBufferFromPtr(args[i])
          }
        });
      }

      compute.setBindGroup(0, this.device.createBindGroup({
        layout: bindGroupLayout,
        entries: bindGroupEntries
      }));
      const wl: Array<number> = [1, 1, 1, 1, 1, 1];
      for (let i = 0; i < dispatchToDim.length; ++i) {
        wl[dispatchToDim[i]] = args[layoutEntries.length + i];
      }
      compute.dispatchWorkgroups(wl[0], wl[1], wl[2]);
      compute.end();
      if (this.isProfilingEnabled()) {
        if (this.curTimeQueryIdx < this.numQueryCount) {
          commandEncoder.writeTimestamp(this.timeQuerySet, this.curTimeQueryIdx + 1);
          this.curTimeQueryIdx = this.curTimeQueryIdx + 2;
        }
      }
      const command = commandEncoder.finish();
      this.device.queue.submit([command]);
    };

    this.shaderFuncMap.set(name, submitShader);

    return submitShader;
  }

  executeAfterCompilation(callbackForSuccess: Function, callbackForFail: Function): void {
    new Promise(resolve => {setTimeout(resolve, 1000)}).then(() => {
      const perf = compact.getPerformance();
      const durationMs = perf.now() - this.compilationStartTime;
      const timeout = 60 * 1000;
      if (this.compilationStatus == ShaderModuleCompilationStatus.compiling && durationMs <= timeout) {
        this.log("Compiling: " + this.compilationInfo);
        this.executeAfterCompilation(callbackForSuccess, callbackForFail);
      } else {
        //this.log("compilationStatus " + compilationStatus);
        if (this.compilationStatus == ShaderModuleCompilationStatus.successful) {
          this.log("Compilation success: " + this.compilationInfo);
          callbackForSuccess();
        } else {
          this.log("Compilation failed: " + this.compilationInfo);
          callbackForFail();
        }
      }
    });
  }

  /**
   * Get the device API according to its name
   * @param The name of the API.
   * @returns The corresponding device api.
   */
  getDeviceAPI(name: string): Function {
    if (name == "deviceAllocDataSpace") {
      return (nbytes: number): GPUPointer => {
        return this.deviceAllocDataSpace(nbytes);
      };
    } else if (name == "deviceFreeDataSpace") {
      return (ptr: GPUPointer): void => {
        return this.deviceFreeDataSpace(ptr);
      };
    } else if (name == "deviceCopyToGPU") {
      return (
        from: Pointer,
        to: GPUPointer,
        toOffset: number,
        nbytes: number
      ): void => {
        this.deviceCopyToGPU(from, to, toOffset, nbytes);
      };
    } else if (name == "deviceCopyFromGPU") {
      return (
        from: GPUPointer,
        fromOffset: number,
        to: Pointer,
        nbytes: number
      ): void => {
        this.deviceCopyFromGPU(from, fromOffset, to, nbytes);
      };
    } else if (name == "deviceCopyWithinGPU") {
      return (
        from: GPUPointer,
        fromOffset: number,
        to: Pointer,
        toOffset: number,
        nbytes: number
      ): void => {
        this.deviceCopyWithinGPU(from, fromOffset, to, toOffset, nbytes);
      };
    } else {
      throw new Error("Unknown DeviceAPI function " + name);
    }

  }

  // DeviceAPI
  private deviceAllocDataSpace(nbytes: number): GPUPointer {
    const buffer = this.device.createBuffer({
      size: nbytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    return this.attachToBufferTable(buffer, nbytes);
  }

  private deviceAllocDataSpaceForQueryResolve(nbytes: number): GPUPointer {
    const buffer = this.device.createBuffer({
      size: nbytes,
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.QUERY_RESOLVE,
    });
    return this.attachToBufferTable(buffer, nbytes);
  }

  private deviceAllocDataSpaceForQueryResolveResult(nbytes: number): GPUPointer {
    const buffer = this.device.createBuffer({
      size: nbytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    return this.attachToBufferTable(buffer, nbytes);
  }

  private deviceFreeDataSpace(ptr: GPUPointer): void {
    const idx = ptr;
    const buffer = this.bufferTable[idx];
    this.bufferTable[idx] = undefined;
    this.bufferSizeTable[idx] = 0;
    assert(buffer !== undefined);
    this.bufferTableFreeId.push(idx);
    buffer.destroy();
  }

  private deviceCopyToGPU(
    from: Pointer,
    to: GPUPointer,
    toOffset: number,
    nbytes: number
  ): void {
    // Perhaps it would be more useful to use a staging buffer?
    const gpuTemp = this.device.createBuffer({
      mappedAtCreation: true,
      size: nbytes,
      usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC
    });

    const cpuTemp = gpuTemp.getMappedRange();

    const viewU8 = new Uint8Array(cpuTemp);
    viewU8.set(this.memory.loadRawBytes(from, nbytes));
    gpuTemp.unmap();

    const copyEncoder = this.device.createCommandEncoder();
    copyEncoder.copyBufferToBuffer(
      gpuTemp,
      0,
      this.gpuBufferFromPtr(to),
      toOffset,
      nbytes
    );
    const copyCommands = copyEncoder.finish();
    this.device.queue.submit([copyCommands]);
    gpuTemp.destroy();
  }

  private deviceCopyFromGPU(
    from: GPUPointer,
    fromOffset: number,
    to: Pointer,
    nbytes: number
  ): void {
    // Perhaps it would be more useful to resuse a staging buffer?
    const gpuTemp = this.device.createBuffer({
      size: nbytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const copyEncoder = this.device.createCommandEncoder();
    copyEncoder.copyBufferToBuffer(
      this.gpuBufferFromPtr(from),
      fromOffset,
      gpuTemp,
      0,
      nbytes
    );
    const copyCommands = copyEncoder.finish();
    this.device.queue.submit([copyCommands]);

    this.numPendingReads += 1;

    const readEvent = gpuTemp.mapAsync(GPUMapMode.READ).then((data: unknown) => {
      this.memory.storeRawBytes(to, new Uint8Array(data as ArrayBuffer));
      this.numPendingReads -= 1;
      gpuTemp.destroy();
    });

    if (this.numPendingReads == 1) {
      this.pendingRead = readEvent;
    } else {
      this.pendingRead = Promise.all([
        this.pendingRead,
        readEvent,
        // eslint-disable-next-line @typescript-eslint/no-empty-function
      ]).then(() => {});
    }
  }

  private deviceCopyWithinGPU(
    from: GPUPointer,
    fromOffset: number,
    to: Pointer,
    toOffset: number,
    nbytes: number
  ): void {
    const copyEncoder = this.device.createCommandEncoder();
    copyEncoder.copyBufferToBuffer(
      this.gpuBufferFromPtr(from),
      fromOffset,
      this.gpuBufferFromPtr(to),
      toOffset,
      nbytes
    );
    const copyCommands = copyEncoder.finish();
    this.device.queue.submit([copyCommands]);
  }

  private gpuBufferFromPtr(ptr: GPUPointer): GPUBuffer {
    const buffer = this.bufferTable[ptr];
    assert(buffer !== undefined);
    return buffer;
  }

  private attachToBufferTable(buffer: GPUBuffer, nbytes: number): GPUPointer {
    if (this.bufferTableFreeId.length != 0) {
      const idx = this.bufferTableFreeId.pop() as number;
      this.bufferTable[idx] = buffer;
      this.bufferSizeTable[idx] = nbytes;
      return idx;
    } else {
      const idx = this.bufferTable.length;
      this.bufferTable.push(buffer);
      this.bufferSizeTable.push(nbytes);
      return idx;
    }
  }
}
