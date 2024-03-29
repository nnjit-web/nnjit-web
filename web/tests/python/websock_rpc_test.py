# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Simple testcode to test Javascript RPC

To use it, start a rpc proxy with "python -m tvm.exec.rpc_proxy".
Connect javascript end to the websocket port and connect to the RPC.
"""

import os
import time
import tvm
from tvm import te
from tvm import rpc
from tvm.contrib import utils, emcc
from tvm.relay.backend import Runtime
import numpy as np
from kernel_modules.common import get_func_mod

proxy_host = "127.0.0.1"
proxy_port = 9090
session_timeout = 10


class AddOne:
    def __init__(self):
        super().__init__()
        self._mod_name = "addone"
    
    
    def build(self, target, runtime):
        n = te.var("n")
        self._A = A = te.placeholder((n,), name="A")
        B = te.compute(A.shape, lambda *i: A(*i) + 1.0, name="B")
        s = te.create_schedule(B.op)

        mod_name = "addone"
        fadd = tvm.build(s, [A, B], target, runtime=runtime, name=mod_name)
        print(tvm.lower(s, [A, B], simple_mode=True))
        
        build_rounds = 1
        start_time = time.time()
        for i in range(build_rounds - 1):
            print("Build %d/%d" % (i + 1, build_rounds))
            fadd = tvm.build(s, [A, B], target, runtime=runtime, name=mod_name)
        total_time = time.time() - start_time
        print("Build time: total %.6f s, avg %.6f s" % (total_time, total_time / build_rounds))
        return fadd
    

    def test_rpc(self, remote_system_lib, dev):
        dtype = self._A.dtype
        a = tvm.nd.array(np.random.uniform(size=1024).astype(dtype), dev)
        b = tvm.nd.array(np.zeros(1024, dtype=dtype), dev)
        # invoke the function
        addone = remote_system_lib.get_function("addone")
        addone(a, b)
        np.testing.assert_equal(b.numpy(), a.numpy() + 1)
        print("Test pass!")


    def test_time_evaluation(self, remote_system_lib, dev):
        dtype = self._A.dtype
        a = tvm.nd.array(np.random.uniform(size=1024).astype(dtype), dev)
        b = tvm.nd.array(np.zeros(1024, dtype=dtype), dev)
        # time evaluator
        time_f = remote_system_lib.time_evaluator(self._mod_name, dev, number=100, repeat=10)
        time_f(a, b)
        cost = time_f(a, b).mean
        print("%g secs/op" % cost)
        np.testing.assert_equal(b.numpy(), a.numpy() + 1)


def test_rpc():
    if not tvm.runtime.enabled("rpc"):
        return
    # generate the wasm library
    runtime = Runtime("cpp", {"system-lib": True})
    target = "llvm -mtriple=wasm32-unknown-unknown-wasm"
    use_simd = os.getenv("EMCC_USE_SIMD")
    use_simd = (use_simd is not None) and (use_simd == "1")
    if use_simd:
        target = target + " -mattr=+simd128"
    if not tvm.runtime.enabled(target):
        raise RuntimeError("Target %s is not enbaled" % target)
    
    mod_name = os.getenv("TEST_MODULE_NAME")
    print("Module name:", mod_name)
    func_mod = get_func_mod(mod_name)

    mod = func_mod.build(target, runtime)

    enable_export_wasm = True
    if not enable_export_wasm:
        return

    use_ramdisk = False
    if use_ramdisk:
        wasm_path = "/mnt/ramdisk/%s_llvm.wasm" % mod_name
    else:
        #temp = utils.tempdir()
        #wasm_path = temp.relpath("%s.wasm" % mod_name)
        wasm_path = "dist/wasm/%s_llvm.wasm" % mod_name

    #mod.export_library(wasm_path, emcc.create_tvmjs_wasm)

    export_rounds = 1
    start_time = time.time()
    for i in range(export_rounds):
        #print("Export %d/%d" % (i + 1, export_rounds))
        mod.export_library(wasm_path, emcc.create_tvmjs_wasm)
    total_time = time.time() - start_time
    print("Export library: rounds %d, total %.6f s, avg %.6f s" % (
            export_rounds, total_time, total_time / export_rounds))

    enable_websock_rpc_test = True
    if not enable_websock_rpc_test:
        return

    wasm_binary = open(wasm_path, "rb").read()
    key = "wasm"
    remote = rpc.connect(
        proxy_host,
        proxy_port,
        key=key,
        session_timeout=session_timeout,
        session_constructor_args=["rpc.WasmSession", wasm_binary],
    )

    def check(remote):
        '''
        # basic function checks.
        faddone = remote.get_function("testing.asyncAddOne")
        fecho = remote.get_function("testing.echo")
        assert faddone(100) == 101
        assert fecho(1, 2, 3) == 1
        assert fecho(1, 2, 3) == 1
        assert fecho(100, 2, 3) == 100
        assert fecho("xyz") == "xyz"
        assert bytes(fecho(bytearray(b"123"))) == b"123"
        '''

        # run the generated library.
        f1 = remote.system_lib()
        dev = remote.cpu(0)
        #func_mod.test_rpc(f1, dev)
        func_mod.test_time_evaluation(remote, dev)

    check(remote)


test_rpc()
