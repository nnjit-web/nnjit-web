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
Object.defineProperty(exports, "__esModule", { value: true });
exports.assert = exports.detectGPUDevice = exports.wasmPath = exports.RPCServer = exports.instantiate = exports.Instance = exports.NDArray = exports.Module = exports.DLDataType = exports.DLDevice = exports.Scalar = void 0;
var runtime_1 = require("./runtime");
Object.defineProperty(exports, "Scalar", { enumerable: true, get: function () { return runtime_1.Scalar; } });
Object.defineProperty(exports, "DLDevice", { enumerable: true, get: function () { return runtime_1.DLDevice; } });
Object.defineProperty(exports, "DLDataType", { enumerable: true, get: function () { return runtime_1.DLDataType; } });
Object.defineProperty(exports, "Module", { enumerable: true, get: function () { return runtime_1.Module; } });
Object.defineProperty(exports, "NDArray", { enumerable: true, get: function () { return runtime_1.NDArray; } });
Object.defineProperty(exports, "Instance", { enumerable: true, get: function () { return runtime_1.Instance; } });
Object.defineProperty(exports, "instantiate", { enumerable: true, get: function () { return runtime_1.instantiate; } });
var rpc_server_1 = require("./rpc_server");
Object.defineProperty(exports, "RPCServer", { enumerable: true, get: function () { return rpc_server_1.RPCServer; } });
var support_1 = require("./support");
Object.defineProperty(exports, "wasmPath", { enumerable: true, get: function () { return support_1.wasmPath; } });
var webgpu_1 = require("./webgpu");
Object.defineProperty(exports, "detectGPUDevice", { enumerable: true, get: function () { return webgpu_1.detectGPUDevice; } });
var support_2 = require("./support");
Object.defineProperty(exports, "assert", { enumerable: true, get: function () { return support_2.assert; } });
//# sourceMappingURL=index.js.map