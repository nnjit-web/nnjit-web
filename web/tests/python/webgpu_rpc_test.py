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
import tvm
from tvm import te
from tvm import rpc
from tvm.contrib import utils, emcc
from tvm.relay.backend import Runtime
import numpy as np
from kernel_modules.common import get_func_mod

proxy_host = "127.0.0.1"
proxy_port = 9090
proxy_key = "hp-elitedesk-800-g6"
session_timeout = 10


def test_rpc():
    if not tvm.runtime.enabled("rpc"):
        return
    # generate the wasm library
    target_host = "llvm -mtriple=wasm32-unknown-unknown-wasm"
    #target = tvm.target.Target("webgpu", host=target_host)
    target = "webgpu"
    #target = "vulkan"
    #target = {"kind": "vulkan", "keys": ["webgpu", "gpu"]}
    target = tvm.target.Target(target, host=target_host)
    print("target", target)
    runtime = Runtime("cpp", {"system-lib": True})
    if not tvm.runtime.enabled(target_host):
        raise RuntimeError("Target %s is not enbaled" % target_host)

    n = 2048
    A = te.placeholder((n,), name="A")
    B = te.compute(A.shape, lambda *i: A(*i) + 1.0, name="B")
    s = te.create_schedule(B.op)

    num_thread = 2
    xo, xi = s[B].split(B.op.axis[0], factor=num_thread)
    s[B].bind(xi, te.thread_axis("threadIdx.x"))
    s[B].bind(xo, te.thread_axis("blockIdx.x"))

    print(tvm.lower(s, [A, B], simple_mode=True))
    fadd = tvm.build(s, [A, B], target, runtime=runtime, name="addone")
    temp = utils.tempdir()

    #wasm_path = temp.relpath("addone_gpu.wasm")
    wasm_path = "dist/wasm/%s.wasm" % "addone_gpu"
    fadd.export_library(wasm_path, emcc.create_tvmjs_wasm)

    wasm_binary = open(wasm_path, "rb").read()
    remote = rpc.connect(
        proxy_host,
        proxy_port,
        key=proxy_key,
        session_timeout=session_timeout,
        session_constructor_args=["rpc.WasmSession", wasm_binary],
    )

    def check(remote, func_name):
        # basic function checks.
        dev = remote.webgpu(0)
        #dev = remote.vulkan(0)
        adata = np.random.uniform(size=n).astype(A.dtype)
        a = tvm.nd.array(adata, dev)
        b = tvm.nd.array(np.zeros(n, dtype=A.dtype), dev)
        c = tvm.nd.array(np.zeros(n, dtype=A.dtype), remote.cpu(0))

        def check_call():
            register_webgpu_kernel_func_f = remote.get_function("__sync.wasm.registerWebGPUKernelFunc")
            wait_f = remote.get_function("__async.wasm.WebGPUWaitForTasks")
            
            register_webgpu_kernel_func_f(func_name)
            #np.testing.assert_equal(a.numpy(), adata)
            f1 = remote.system_lib()
            kernel_func = f1.get_function(func_name)
            kernel_func(a, b)
            #remote_adata = a.numpy()
            #remote_bdata = b.numpy()
            #b.copyto(c)
            wait_f()
            #print("adata", adata)
            #print("remote_adata", remote_adata)
            #print("remote_bdata", remote_bdata)
            #np.testing.assert_equal(b.numpy(), a.numpy() + 1)

        #return

        def check_time_evaluation():
            #time_f = f1.time_evaluator("addone", dev, number=100, repeat=10)
            #time_f(a, b)
            #cost = time_f(a, b).mean

            f1 = remote
            time_f = f1.get_function("__sync.wasm.TimeExecutionForWebGPU")
            is_finished_f = f1.get_function("__sync.wasm.isTimeExecutionForWebGPUFinished")
            get_ret_f = f1.get_function("__sync.wasm.getTimeExecutionForWebGPUResults")
            num_repeat = 64
            time_f(func_name, dev, num_repeat, a, b)
            while is_finished_f() == 0:
                import time
                time.sleep(1)
            cost_bytes = get_ret_f()
            cost_arr = np.frombuffer(cost_bytes, dtype=np.float64)
            cost = cost_arr.mean()
            print("%g secs/op" % cost)
        
        #check_call()
        check_time_evaluation()

        print("Test pass..")

    check(remote, "addone")


def test_rpc_v2():
    if not tvm.runtime.enabled("rpc"):
        return
    
    target_host = "llvm -mtriple=wasm32-unknown-unknown-wasm"
    target = "webgpu"
    target = tvm.target.Target(target, host=target_host)
    print("target", target)
    runtime = Runtime("cpp", {"system-lib": True})
    if not tvm.runtime.enabled(target_host):
        raise RuntimeError("Target %s is not enbaled" % target_host)
    
    mod_name = os.getenv("TEST_MODULE_NAME")
    print("Module name:", mod_name)
    func_mod = get_func_mod(mod_name)

    mod = func_mod.build(target, runtime)

    wasm_path = "dist/wasm/%s.wasm" % mod_name
    mod.export_library(wasm_path, emcc.create_tvmjs_wasm)

    wasm_binary = open(wasm_path, "rb").read()
    remote = rpc.connect(
        proxy_host,
        proxy_port,
        key=proxy_key,
        session_timeout=session_timeout,
        session_constructor_args=["rpc.WasmSession", wasm_binary],
    )

    def check(remote):
        f1 = remote
        dev = remote.webgpu(0)
        #func_mod.test_rpc(f1, dev)
        func_mod.test_time_evaluation(f1, dev)

    check(remote)


#test_rpc()
test_rpc_v2()
