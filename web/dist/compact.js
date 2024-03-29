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
/** NodeJS and Web compact layer */
Object.defineProperty(exports, "__esModule", { value: true });
exports.createWebSocket = exports.getPerformance = void 0;
/**
 * Get performance measurement.
 */
function getPerformance() {
    if (typeof performance == "undefined") {
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        const performanceNode = require("perf_hooks");
        return performanceNode.performance;
    }
    else {
        return performance;
    }
}
exports.getPerformance = getPerformance;
/**
 * Create a new websocket for a given URL
 * @param url The url.
 */
function createWebSocket(url) {
    if (typeof WebSocket == "undefined") {
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        const WebSocket = require("ws");
        return new WebSocket(url);
    }
    else {
        return new WebSocket(url);
    }
}
exports.createWebSocket = createWebSocket;
//# sourceMappingURL=compact.js.map