"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.WebGPUContext = exports.detectGPUDevice = void 0;
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
const support_1 = require("./support");
const compact = require("./compact");
/**
 * DetectGPU device in the environment.
 */
function detectGPUDevice() {
    return __awaiter(this, void 0, void 0, function* () {
        if (typeof navigator !== "undefined" && navigator.gpu !== undefined) {
            const adapter = yield navigator.gpu.requestAdapter();
            const supportTimeQuery = adapter === null || adapter === void 0 ? void 0 : adapter.features.has("timestamp-query");
            const deviceDescriptor = {};
            if (supportTimeQuery) {
                deviceDescriptor.requiredFeatures = ["timestamp-query"];
            }
            return yield (adapter === null || adapter === void 0 ? void 0 : adapter.requestDevice(deviceDescriptor));
        }
        else {
            return undefined;
        }
    });
}
exports.detectGPUDevice = detectGPUDevice;
var ShaderModuleCompilationStatus;
(function (ShaderModuleCompilationStatus) {
    ShaderModuleCompilationStatus[ShaderModuleCompilationStatus["none"] = 0] = "none";
    ShaderModuleCompilationStatus[ShaderModuleCompilationStatus["compiling"] = 1] = "compiling";
    ShaderModuleCompilationStatus[ShaderModuleCompilationStatus["failed"] = 2] = "failed";
    ShaderModuleCompilationStatus[ShaderModuleCompilationStatus["successful"] = 3] = "successful";
})(ShaderModuleCompilationStatus || (ShaderModuleCompilationStatus = {}));
/**
 * WebGPU context
 * Manages all the webgpu resources here.
 */
class WebGPUContext {
    constructor(memory, device) {
        //private readBuffer:;
        this.bufferTable = [undefined];
        this.bufferTableFreeId = [];
        this.bufferSizeTable = [];
        this.pendingRead = Promise.resolve();
        this.numPendingReads = 0;
        this.shaderNames = [];
        this.shaderFuncMap = new Map();
        this.isProfilingSupportedFlag = false;
        this.enableProfilingFlag = false;
        this.numQueryCount = 128;
        this.curTimeQueryIdx = 0;
        this.timestampBufferSize = -1;
        this.timestampBufferIdx = -1;
        this.timestampOutBufferIdx = -1;
        this.compilationStartTime = 0;
        this.compilationStatus = ShaderModuleCompilationStatus.none;
        this.compilationInfo = "";
        this.memory = memory;
        this.device = device;
        this.pipeline = undefined;
        try {
            this.timeQuerySet = this.device.createQuerySet({
                type: "timestamp", count: this.numQueryCount
            });
            this.isProfilingSupportedFlag = true;
        }
        catch (err) {
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
    resolveQuerySet() {
        return __awaiter(this, void 0, void 0, function* () {
            const timeQueryCount = this.curTimeQueryIdx;
            const commandEncoder = this.device.createCommandEncoder();
            for (let i = 0; i < timeQueryCount; i += 32) {
                commandEncoder.resolveQuerySet(this.timeQuerySet, i, 32, this.gpuBufferFromPtr(this.timestampBufferIdx), (i / 32) * 256);
            }
            const command = commandEncoder.finish();
            this.device.queue.submit([command]);
            this.deviceCopyWithinGPU(this.timestampBufferIdx, 0, this.timestampOutBufferIdx, 0, this.timestampBufferSize);
            yield this.sync();
        });
    }
    getTimeCostArray(unit, outArr) {
        return __awaiter(this, void 0, void 0, function* () {
            if (this.isProfilingEnabled() == false) {
                return;
            }
            if (this.timestampBufferIdx >= 0) {
                yield this.resolveQuerySet();
                const timestampBuffer = this.gpuBufferFromPtr(this.timestampOutBufferIdx);
                yield timestampBuffer.mapAsync(GPUMapMode.READ);
                const cpuTemp = timestampBuffer.getMappedRange();
                const viewU64 = new BigUint64Array(cpuTemp);
                for (let i = 0; i < this.curTimeQueryIdx; i += 2) {
                    const durationNs = viewU64[i + 1] - viewU64[i];
                    let speed = Number(durationNs);
                    if (unit == "s") {
                        speed = speed / 1e9;
                    }
                    else if (unit == "ms") {
                        speed = speed / 1e6;
                    }
                    else if (unit == "us") {
                        speed = speed / 1e3;
                    }
                    outArr.push(speed);
                }
                timestampBuffer.unmap();
            }
        });
    }
    log(msg) {
        this.logger("WebGPUContext: " + msg);
    }
    getShaderNames() {
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
    sync() {
        return __awaiter(this, void 0, void 0, function* () {
            return this.device.queue.onSubmittedWorkDone();
        });
    }
    /**
     * Create a PackedFunc that runs the given shader
     *
     * @param info The function information in json.
     * @param data The shader data(in SPIRV)
     */
    createShader(name, info, data) {
        if (!this.shaderNames.includes(name)) {
            this.shaderNames.push(name);
        }
        if (this.shaderFuncMap.get(name) != undefined) {
            return this.shaderFuncMap.get(name);
        }
        //if (!name.includes("default_function_kernel") && !name.includes("batch_matmul")) {
        //  const submitShader = (...args: Array<GPUPointer | number>): void => {};
        //  this.shaderFuncMap.set(name, submitShader);
        //  return submitShader;
        //}
        const finfo = JSON.parse(info);
        const layoutEntries = [];
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
            }
            else {
                throw new Error("Cannot handle argument type " + dtype + " in WebGPU shader");
            }
        }
        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: layoutEntries
        });
        const getDateString = () => {
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
            const submitNothing = (...args) => {
            };
            return submitNothing;
        }
        const srcCode = new Uint32Array(data.buffer);
        //const srcCode = String.fromCharCode.apply(null, data);
        //const srcCode = new TextDecoder().decode(data);
        const perf = compact.getPerformance();
        let tstart = perf.now();
        const shaderModule = this.device.createShaderModule({
            code: srcCode
        });
        let tend = perf.now();
        let durationMs = tend - tstart;
        this.log("createShaderModule " + durationMs + " ms");
        this.device.pushErrorScope("validation");
        const timeout = 60 * 1000; // ms
        this.compilationStartTime = tstart = perf.now();
        let pipeline = undefined;
        let enableAsync = true;
        if (!name.includes("default_function_kernel")) {
            enableAsync = false;
        }
        if (!enableAsync) {
            pipeline = this.device.createComputePipeline({
                layout: this.device.createPipelineLayout({
                    bindGroupLayouts: [bindGroupLayout]
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
                }
                else {
                    this.compilationStatus = ShaderModuleCompilationStatus.successful;
                }
            });
        }
        else {
            this.device.createComputePipelineAsync({
                layout: this.device.createPipelineLayout({
                    bindGroupLayouts: [bindGroupLayout]
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
                }
                else {
                    pipeline = outputPipeline;
                    this.device.popErrorScope().then((error) => {
                        if (error) {
                            this.compilationStatus = ShaderModuleCompilationStatus.failed;
                            this.log("createComputePipeline: error");
                            if (error instanceof GPUValidationError) {
                                this.log("Error: " + error.message);
                            }
                        }
                        else {
                            this.compilationStatus = ShaderModuleCompilationStatus.successful;
                        }
                    });
                }
            }).catch((error) => {
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
        const dispatchToDim = [];
        for (let i = 0; i < finfo.launch_param_tags.length; ++i) {
            const tag = finfo.launch_param_tags[i];
            if (tag.startsWith("blockIdx.")) {
                const target = tag.charCodeAt(tag.length - 1) - ("x".charCodeAt(0));
                (0, support_1.assert)(target >= 0 && target < 3);
                dispatchToDim.push(target);
            }
            else if (tag.startsWith("threadIdx.")) {
                const target = tag.charCodeAt(tag.length - 1) - ("x".charCodeAt(0));
                (0, support_1.assert)(target >= 0 && target < 3);
                dispatchToDim.push(target + 3);
            }
            else {
                throw new Error("Cannot handle thread_axis " + tag);
            }
        }
        //tend = perf.now();
        //durationMs = tend - tstart;
        //this.log("setDispatchToDim " + durationMs + " ms");
        const submitShader = (...args) => {
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
            const bindGroupEntries = [];
            (0, support_1.assert)(args.length == layoutEntries.length + dispatchToDim.length);
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
            const wl = [1, 1, 1, 1, 1, 1];
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
    executeAfterCompilation(callbackForSuccess, callbackForFail) {
        new Promise(resolve => { setTimeout(resolve, 1000); }).then(() => {
            const perf = compact.getPerformance();
            const durationMs = perf.now() - this.compilationStartTime;
            const timeout = 60 * 1000;
            if (this.compilationStatus == ShaderModuleCompilationStatus.compiling && durationMs <= timeout) {
                this.log("Compiling: " + this.compilationInfo);
                this.executeAfterCompilation(callbackForSuccess, callbackForFail);
            }
            else {
                //this.log("compilationStatus " + compilationStatus);
                if (this.compilationStatus == ShaderModuleCompilationStatus.successful) {
                    this.log("Compilation success: " + this.compilationInfo);
                    callbackForSuccess();
                }
                else {
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
    getDeviceAPI(name) {
        if (name == "deviceAllocDataSpace") {
            return (nbytes) => {
                return this.deviceAllocDataSpace(nbytes);
            };
        }
        else if (name == "deviceFreeDataSpace") {
            return (ptr) => {
                return this.deviceFreeDataSpace(ptr);
            };
        }
        else if (name == "deviceCopyToGPU") {
            return (from, to, toOffset, nbytes) => {
                this.deviceCopyToGPU(from, to, toOffset, nbytes);
            };
        }
        else if (name == "deviceCopyFromGPU") {
            return (from, fromOffset, to, nbytes) => {
                this.deviceCopyFromGPU(from, fromOffset, to, nbytes);
            };
        }
        else if (name == "deviceCopyWithinGPU") {
            return (from, fromOffset, to, toOffset, nbytes) => {
                this.deviceCopyWithinGPU(from, fromOffset, to, toOffset, nbytes);
            };
        }
        else {
            throw new Error("Unknown DeviceAPI function " + name);
        }
    }
    // DeviceAPI
    deviceAllocDataSpace(nbytes) {
        const buffer = this.device.createBuffer({
            size: nbytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        return this.attachToBufferTable(buffer, nbytes);
    }
    deviceAllocDataSpaceForQueryResolve(nbytes) {
        const buffer = this.device.createBuffer({
            size: nbytes,
            usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.QUERY_RESOLVE,
        });
        return this.attachToBufferTable(buffer, nbytes);
    }
    deviceAllocDataSpaceForQueryResolveResult(nbytes) {
        const buffer = this.device.createBuffer({
            size: nbytes,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        return this.attachToBufferTable(buffer, nbytes);
    }
    deviceFreeDataSpace(ptr) {
        const idx = ptr;
        const buffer = this.bufferTable[idx];
        this.bufferTable[idx] = undefined;
        this.bufferSizeTable[idx] = 0;
        (0, support_1.assert)(buffer !== undefined);
        this.bufferTableFreeId.push(idx);
        buffer.destroy();
    }
    deviceCopyToGPU(from, to, toOffset, nbytes) {
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
        copyEncoder.copyBufferToBuffer(gpuTemp, 0, this.gpuBufferFromPtr(to), toOffset, nbytes);
        const copyCommands = copyEncoder.finish();
        this.device.queue.submit([copyCommands]);
        gpuTemp.destroy();
    }
    deviceCopyFromGPU(from, fromOffset, to, nbytes) {
        // Perhaps it would be more useful to resuse a staging buffer?
        const gpuTemp = this.device.createBuffer({
            size: nbytes,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        const copyEncoder = this.device.createCommandEncoder();
        copyEncoder.copyBufferToBuffer(this.gpuBufferFromPtr(from), fromOffset, gpuTemp, 0, nbytes);
        const copyCommands = copyEncoder.finish();
        this.device.queue.submit([copyCommands]);
        this.numPendingReads += 1;
        const readEvent = gpuTemp.mapAsync(GPUMapMode.READ).then((data) => {
            this.memory.storeRawBytes(to, new Uint8Array(data));
            this.numPendingReads -= 1;
            gpuTemp.destroy();
        });
        if (this.numPendingReads == 1) {
            this.pendingRead = readEvent;
        }
        else {
            this.pendingRead = Promise.all([
                this.pendingRead,
                readEvent,
                // eslint-disable-next-line @typescript-eslint/no-empty-function
            ]).then(() => { });
        }
    }
    deviceCopyWithinGPU(from, fromOffset, to, toOffset, nbytes) {
        const copyEncoder = this.device.createCommandEncoder();
        copyEncoder.copyBufferToBuffer(this.gpuBufferFromPtr(from), fromOffset, this.gpuBufferFromPtr(to), toOffset, nbytes);
        const copyCommands = copyEncoder.finish();
        this.device.queue.submit([copyCommands]);
    }
    gpuBufferFromPtr(ptr) {
        const buffer = this.bufferTable[ptr];
        (0, support_1.assert)(buffer !== undefined);
        return buffer;
    }
    attachToBufferTable(buffer, nbytes) {
        if (this.bufferTableFreeId.length != 0) {
            const idx = this.bufferTableFreeId.pop();
            this.bufferTable[idx] = buffer;
            this.bufferSizeTable[idx] = nbytes;
            return idx;
        }
        else {
            const idx = this.bufferTable.length;
            this.bufferTable.push(buffer);
            this.bufferSizeTable.push(nbytes);
            return idx;
        }
    }
}
exports.WebGPUContext = WebGPUContext;
//# sourceMappingURL=webgpu.js.map
