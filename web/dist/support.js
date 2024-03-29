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
exports.wasmPath = exports.assert = exports.Uint8ArrayToString = exports.StringToUint8Array = void 0;
/**
 * Convert string to Uint8array.
 * @param str The string.
 * @returns The corresponding Uint8Array.
 */
function StringToUint8Array(str) {
    const arr = new Uint8Array(str.length + 1);
    for (let i = 0; i < str.length; ++i) {
        arr[i] = str.charCodeAt(i);
    }
    arr[str.length] = 0;
    return arr;
}
exports.StringToUint8Array = StringToUint8Array;
/**
 * Convert Uint8array to string.
 * @param array The array.
 * @returns The corresponding string.
 */
function Uint8ArrayToString(arr) {
    const ret = [];
    for (const ch of arr) {
        ret.push(String.fromCharCode(ch));
    }
    return ret.join("");
}
exports.Uint8ArrayToString = Uint8ArrayToString;
/**
 * Internal assert helper
 * @param condition condition The condition to fail.
 * @param msg msg The message.
 */
function assert(condition, msg) {
    if (!condition) {
        throw new Error("AssertError:" + (msg || ""));
    }
}
exports.assert = assert;
/**
 * Get the path to the wasm library in nodejs.
 * @return The wasm path.
 */
function wasmPath() {
    return __dirname + "/wasm";
}
exports.wasmPath = wasmPath;
//# sourceMappingURL=support.js.map