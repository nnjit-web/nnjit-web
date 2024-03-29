
import os
import time
import argparse
import tvm
from tvm import te
from tvm import rpc
from tvm.contrib import utils, wasm
from tvm.relay.backend import Runtime
import numpy as np
from kernel_modules.addone import AddOne
from kernel_modules.matmul import MatMul

proxy_host = "127.0.0.1"
proxy_port = 9090
session_timeout = 10


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mod_name", type=str, required=True, help="Module name")
    return parser.parse_args()


def test_rpc():
    runtime = Runtime("cpp", {"system-lib": True})
    target = "wasm"
    if not tvm.runtime.enabled(target):
        raise RuntimeError("Target %s is not enbaled" % target)

    args = parse_args()
    
    mod_name = args.mod_name
    if mod_name == "addone":
        func_mod = AddOne()
    elif mod_name.startswith("matmul"):
        M = K = N = 64
        #M = K = N = 1024
        func_mod = MatMul(M, K, N, "wasm")
    else:
        raise ValueError("Unsupported module name: " + mod_name)

    mod = func_mod.build(target, runtime)

    enable_export_llbrary = True
    fmt = "wasm"
    wasm_path = "dist/wasm/%s_wasm.%s" % (mod_name, fmt)
    if enable_export_llbrary:
        mod.export_library(wasm_path, wasm.create_wasm)

        export_rounds = 0
        if export_rounds > 0:
            start_time = time.time()
            for i in range(export_rounds):
                #print("Export %d/%d" % (i + 1, export_rounds))
                mod.export_library(wasm_path, wasm.create_wasm)
            total_time = time.time() - start_time
            print("Export library: rounds %d, total %.6f s, avg %.6f s" % (
                    export_rounds, total_time, total_time / export_rounds))

    enable_wasm_rpc_test = True
    if not enable_wasm_rpc_test or fmt == "wat":
        return

    time.sleep(3)

    wasm_binary = open(wasm_path, "rb").read()

    #key = "wasm"
    key = "honor-magicbook-16"

    remote = rpc.connect(
        proxy_host,
        proxy_port,
        key=key,
        session_timeout=session_timeout,
        session_constructor_args=["rpc.WasmSession", wasm_binary],
    )

    def check(remote):
        f1 = remote.system_lib()
        dev = remote.cpu(0)
        func_mod.test_rpc(f1, dev)
        #func_mod.test_time_evaluation(remote, dev)

    check(remote)


test_rpc()
