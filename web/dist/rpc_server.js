"use strict";
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
exports.RPCServer = void 0;
const support_1 = require("./support");
const webgpu_1 = require("./webgpu");
const compact = require("./compact");
const runtime = require("./runtime");
var RPCServerState;
(function (RPCServerState) {
    RPCServerState[RPCServerState["InitHeader"] = 0] = "InitHeader";
    RPCServerState[RPCServerState["InitHeaderKey"] = 1] = "InitHeaderKey";
    RPCServerState[RPCServerState["InitServer"] = 2] = "InitServer";
    RPCServerState[RPCServerState["WaitForCallback"] = 3] = "WaitForCallback";
    RPCServerState[RPCServerState["ReceivePacketHeader"] = 4] = "ReceivePacketHeader";
    RPCServerState[RPCServerState["ReceivePacketBody"] = 5] = "ReceivePacketBody";
})(RPCServerState || (RPCServerState = {}));
/** RPC magic header */
const RPC_MAGIC = 0xff271;
/**
 * An utility class to read from binary bytes.
 */
class ByteStreamReader {
    constructor(bytes) {
        this.offset = 0;
        this.bytes = bytes;
    }
    readU32() {
        const i = this.offset;
        const b = this.bytes;
        const val = b[i] | (b[i + 1] << 8) | (b[i + 2] << 16) | (b[i + 3] << 24);
        this.offset += 4;
        return val;
    }
    readU64() {
        const val = this.readU32();
        this.offset += 4;
        return val;
    }
    readByteArray() {
        const len = this.readU64();
        (0, support_1.assert)(this.offset + len <= this.bytes.byteLength);
        const ret = new Uint8Array(len);
        ret.set(this.bytes.slice(this.offset, this.offset + len));
        this.offset += len;
        return ret;
    }
}
/**
 * A websocket based RPC
 */
class RPCServer {
    constructor(url, key, getImports, logger = console.log) {
        this.state = RPCServerState.InitHeader;
        this.pendingSend = Promise.resolve();
        this.inst = undefined;
        this.currPacketLength = 0;
        this.remoteKeyLength = 0;
        this.pendingBytes = 0;
        this.buffredBytes = 0;
        this.messageQueue = [];
        this.url = url;
        this.key = key;
        this.name = "WebSocketRPCServer[" + this.key + "]: ";
        this.getImports = getImports;
        this.logger = logger;
        this.checkLittleEndian();
        this.socket = compact.createWebSocket(url);
        this.socket.binaryType = "arraybuffer";
        this.socket.addEventListener("open", (event) => {
            return this.onOpen(event);
        });
        this.socket.addEventListener("message", (event) => {
            return this.onMessage(event);
        });
        this.socket.addEventListener("close", (event) => {
            return this.onClose(event);
        });
    }
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    onClose(_event) {
        if (this.inst !== undefined) {
            this.inst.dispose();
        }
        if (this.state == RPCServerState.ReceivePacketHeader) {
            this.log("Closing the server in clean state");
            this.log("Automatic reconnecting..");
            new RPCServer(this.url, this.key, this.getImports, this.logger);
        }
        else {
            this.log("Closing the server, final state=" + this.state);
            this.log("Automatic reconnecting..");
            new RPCServer(this.url, this.key, this.getImports, this.logger);
        }
    }
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    onOpen(_event) {
        // Send the headers
        let bkey = (0, support_1.StringToUint8Array)("server:" + this.key);
        bkey = bkey.slice(0, bkey.length - 1);
        const intbuf = new Int32Array(1);
        intbuf[0] = RPC_MAGIC;
        this.socket.send(intbuf);
        intbuf[0] = bkey.length;
        this.socket.send(intbuf);
        this.socket.send(bkey);
        this.log("connected...");
        // request bytes: magic + keylen
        this.requestBytes(4 /* SizeOf.I32 */ + 4 /* SizeOf.I32 */);
        this.state = RPCServerState.InitHeader;
    }
    /** Handler for raw message. */
    onMessage(event) {
        const buffer = event.data;
        this.buffredBytes += buffer.byteLength;
        this.messageQueue.push(new Uint8Array(buffer));
        this.processEvents();
    }
    /** Process ready events. */
    processEvents() {
        while (this.buffredBytes >= this.pendingBytes && this.pendingBytes != 0) {
            this.onDataReady();
        }
    }
    /** State machine to handle each request */
    onDataReady() {
        switch (this.state) {
            case RPCServerState.InitHeader: {
                this.handleInitHeader();
                break;
            }
            case RPCServerState.InitHeaderKey: {
                this.handleInitHeaderKey();
                break;
            }
            case RPCServerState.ReceivePacketHeader: {
                this.currPacketHeader = this.readFromBuffer(8 /* SizeOf.I64 */);
                const reader = new ByteStreamReader(this.currPacketHeader);
                this.currPacketLength = reader.readU64();
                (0, support_1.assert)(this.pendingBytes == 0);
                this.requestBytes(this.currPacketLength);
                this.state = RPCServerState.ReceivePacketBody;
                break;
            }
            case RPCServerState.ReceivePacketBody: {
                const body = this.readFromBuffer(this.currPacketLength);
                (0, support_1.assert)(this.pendingBytes == 0);
                (0, support_1.assert)(this.currPacketHeader !== undefined);
                this.onPacketReady(this.currPacketHeader, body);
                break;
            }
            case RPCServerState.WaitForCallback: {
                (0, support_1.assert)(this.pendingBytes == 0);
                break;
            }
            default: {
                throw new Error("Cannot handle state " + this.state);
            }
        }
    }
    onPacketReady(header, body) {
        if (this.inst === undefined) {
            // initialize server.
            const reader = new ByteStreamReader(body);
            // eslint-disable-next-line @typescript-eslint/no-unused-vars
            const code = reader.readU32();
            // eslint-disable-next-line @typescript-eslint/no-unused-vars
            const ver = (0, support_1.Uint8ArrayToString)(reader.readByteArray());
            const nargs = reader.readU32();
            const tcodes = [];
            const args = [];
            for (let i = 0; i < nargs; ++i) {
                tcodes.push(reader.readU32());
            }
            for (let i = 0; i < nargs; ++i) {
                const tcode = tcodes[i];
                if (tcode == 11 /* ArgTypeCode.TVMStr */) {
                    const str = (0, support_1.Uint8ArrayToString)(reader.readByteArray());
                    args.push(str);
                }
                else if (tcode == 12 /* ArgTypeCode.TVMBytes */) {
                    args.push(reader.readByteArray());
                }
                else {
                    throw new Error("cannot support type code " + tcode);
                }
            }
            this.onInitServer(args, header, body);
        }
        else {
            (0, support_1.assert)(this.serverRecvData !== undefined);
            this.serverRecvData(header, body);
            this.requestBytes(8 /* SizeOf.I64 */);
            this.state = RPCServerState.ReceivePacketHeader;
        }
    }
    /** Event handler during server initialization. */
    onInitServer(args, header, body) {
        // start the server
        (0, support_1.assert)(args[0] == "rpc.WasmSession");
        (0, support_1.assert)(this.pendingBytes == 0);
        const asyncInitServer = () => __awaiter(this, void 0, void 0, function* () {
            var _a;
            (0, support_1.assert)(args[1] instanceof Uint8Array);
            const inst = yield runtime.instantiate(args[1].buffer, this.getImports(), this.logger);
            try {
                const gpuDevice = yield (0, webgpu_1.detectGPUDevice)();
                if (gpuDevice !== undefined && gpuDevice !== null) {
                    const label = ((_a = gpuDevice.label) === null || _a === void 0 ? void 0 : _a.toString()) || "WebGPU";
                    this.log("Initialize GPU device: " + label);
                    inst.initWebGPU(gpuDevice);
                }
            }
            catch (err) {
                this.log("Cannnot initialize WebGPU, " + err.toString());
            }
            this.inst = inst;
            const fcreate = this.inst.getGlobalFunc("rpc.CreateEventDrivenServer");
            const messageHandler = fcreate((cbytes) => {
                (0, support_1.assert)(this.inst !== undefined);
                if (this.socket.readyState == 1) {
                    // WebSocket will automatically close the socket
                    // if we burst send data that exceeds its internal buffer
                    // wait a bit before we send next one.
                    const sendDataWithCongestionControl = () => __awaiter(this, void 0, void 0, function* () {
                        const packetSize = 4 << 10;
                        const maxBufferAmount = 4 * packetSize;
                        const waitTimeMs = 20;
                        for (let offset = 0; offset < cbytes.length; offset += packetSize) {
                            const end = Math.min(offset + packetSize, cbytes.length);
                            while (this.socket.bufferedAmount >= maxBufferAmount) {
                                yield new Promise((r) => setTimeout(r, waitTimeMs));
                            }
                            this.socket.send(cbytes.slice(offset, end));
                        }
                    });
                    // Chain up the pending send so that the async send is always in-order.
                    this.pendingSend = this.pendingSend.then(sendDataWithCongestionControl);
                    // Directly return since the data are "sent" from the caller's pov.
                    return this.inst.scalar(cbytes.length, "int32");
                }
                else {
                    return this.inst.scalar(0, "int32");
                }
            }, this.name, this.key);
            fcreate.dispose();
            const writeFlag = this.inst.scalar(3, "int32");
            this.serverRecvData = (header, body) => {
                if (messageHandler(header, writeFlag) == 0) {
                    this.socket.close();
                }
                if (messageHandler(body, writeFlag) == 0) {
                    this.socket.close();
                }
            };
            // Forward the same init sequence to the wasm RPC.
            // The RPC will look for "rpc.wasmSession"
            // and we will redirect it to the correct local session.
            // register the callback to redirect the session to local.
            const flocal = this.inst.getGlobalFunc("wasm.LocalSession");
            const localSession = flocal();
            flocal.dispose();
            (0, support_1.assert)(localSession instanceof runtime.Module);
            // eslint-disable-next-line @typescript-eslint/no-unused-vars
            this.inst.registerFunc("rpc.WasmSession", 
            // eslint-disable-next-line @typescript-eslint/no-unused-vars
            (_args) => {
                return localSession;
            });
            messageHandler(header, writeFlag);
            messageHandler(body, writeFlag);
            localSession.dispose();
            this.log("Finish initializing the Wasm Server..");
            this.requestBytes(8 /* SizeOf.I64 */);
            this.state = RPCServerState.ReceivePacketHeader;
            // call process events in case there are bufferred data.
            this.processEvents();
        });
        this.state = RPCServerState.WaitForCallback;
        asyncInitServer();
    }
    log(msg) {
        this.logger(this.name + msg);
    }
    handleInitHeader() {
        const reader = new ByteStreamReader(this.readFromBuffer(4 /* SizeOf.I32 */ * 2));
        const magic = reader.readU32();
        if (magic == RPC_MAGIC + 1) {
            throw new Error("key: " + this.key + " has already been used in proxy");
        }
        else if (magic == RPC_MAGIC + 2) {
            throw new Error("RPCProxy do not have matching client key " + this.key);
        }
        (0, support_1.assert)(magic == RPC_MAGIC, this.url + " is not an RPC Proxy");
        this.remoteKeyLength = reader.readU32();
        (0, support_1.assert)(this.pendingBytes == 0);
        this.requestBytes(this.remoteKeyLength);
        this.state = RPCServerState.InitHeaderKey;
    }
    handleInitHeaderKey() {
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        const remoteKey = (0, support_1.Uint8ArrayToString)(this.readFromBuffer(this.remoteKeyLength));
        (0, support_1.assert)(this.pendingBytes == 0);
        this.requestBytes(8 /* SizeOf.I64 */);
        this.state = RPCServerState.ReceivePacketHeader;
    }
    checkLittleEndian() {
        const a = new ArrayBuffer(4);
        const b = new Uint8Array(a);
        const c = new Uint32Array(a);
        b[0] = 0x11;
        b[1] = 0x22;
        b[2] = 0x33;
        b[3] = 0x44;
        (0, support_1.assert)(c[0] === 0x44332211, "RPCServer little endian to work");
    }
    requestBytes(nbytes) {
        this.pendingBytes += nbytes;
    }
    readFromBuffer(nbytes) {
        const ret = new Uint8Array(nbytes);
        let ptr = 0;
        while (ptr < nbytes) {
            (0, support_1.assert)(this.messageQueue.length != 0);
            const nleft = nbytes - ptr;
            if (this.messageQueue[0].byteLength <= nleft) {
                const buffer = this.messageQueue.shift();
                ret.set(buffer, ptr);
                ptr += buffer.byteLength;
            }
            else {
                const buffer = this.messageQueue[0];
                ret.set(buffer.slice(0, nleft), ptr);
                this.messageQueue[0] = buffer.slice(nleft, buffer.byteLength);
                ptr += nleft;
            }
        }
        this.buffredBytes -= nbytes;
        this.pendingBytes -= nbytes;
        return ret;
    }
}
exports.RPCServer = RPCServer;
//# sourceMappingURL=rpc_server.js.map