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
"""Util to invoke emscripten compilers in the system."""
# pylint: disable=invalid-name
import os
import subprocess
from tvm import autotvm
from tvm._ffi.base import py_str
from tvm._ffi.libinfo import find_lib_path
from .utils import get_dir, get_filename


def create_tvmjs_wasm(output, objects, options=None, cc="emcc"):
    """Create wasm that is supposed to run with the tvmjs.

    Parameters
    ----------
    output : str
        The target shared library.

    objects : list
        List of object files.

    options : str
        The additional options.

    cc : str, optional
        The compile string.
    """
    enable_memory_flags = True
    object_format = "bc"
    enable_with_runtime = True
    enable_test = False
    
    cmd = [cc]
    # [-O0, -O1, -O2, -O3 (default), -Os, -Oz]
    opt_level = os.getenv("EMCC_OPT_LEVEL")
    if opt_level is None:
        opt_level = "-O3"
    cmd += [opt_level]
    #cmd += ["-g"]
    cmd += ["-std=c++14"]
    cmd += ["--no-entry"]
    cmd += ["-s", "WASM=1"]
    cmd += ["-s", "ERROR_ON_UNDEFINED_SYMBOLS=0"]
    cmd += ["-s", "STANDALONE_WASM=0"]
    cmd += ["-s", "WASM_BIGINT=1"]
    cmd += ["-s", "ALLOW_MEMORY_GROWTH=1"]
    cmd += ["-s", "DETERMINISTIC=0"]
    cmd += ["-s", "ASSERTIONS=1"]
    cmd += ["-s", "USE_PTHREADS=0"]
    if enable_memory_flags:
        exec_type = os.getenv("TVM_EXEC_TYPE")
        # uncustomized
        mobile_devices = ["pixel-4-xl", "vivo-x30", "honor-70", "mate-20"]
        if (exec_type is not None and exec_type == "kernel") or \
                os.getenv("DEV_INFO") in mobile_devices:
            cmd += ["-s", "INITIAL_MEMORY=64MB"]
            cmd += ["-s", "TOTAL_MEMORY=128MB"]
            cmd += ["-s", "MAXIMUM_MEMORY=448MB"]
            # For multithread.
            #cmd += ["-s", "TOTAL_MEMORY=16MB"]
            #cmd += ["-s", "MAXIMUM_MEMORY=2048MB"]
        else:
            #cmd += ["-s", "INITIAL_MEMORY=1024MB"]
            #cmd += ["-s", "TOTAL_MEMORY=2048MB"]
            #cmd += ["-s", "MAXIMUM_MEMORY=4096MB"]
            cmd += ["-s", "INITIAL_MEMORY=128MB"]
            cmd += ["-s", "TOTAL_MEMORY=256MB"]
            cmd += ["-s", "MAXIMUM_MEMORY=2048MB"]

    use_simd = os.getenv("EMCC_USE_SIMD")
    use_simd = (use_simd is not None) and (use_simd == "1")
    if use_simd:
        #cmd += ["-s", "SIMD=1"]  # Deprecated.
        cmd += ["-msimd128"]
        cmd += ["-mrelaxed-simd"]
    
    cmd += ["-w", "-Wl,--unresolved-symbols=ignore-all", "-Wl,--undefined=symbol"]

    #print("emcc.py: opt_level %s, use_simd %s" % (opt_level, str(use_simd)))

    objects = [objects] if isinstance(objects, str) else objects

    if enable_with_runtime:
        with_runtime = False
        for obj in objects:
            if obj.find("wasm_runtime.%s" % object_format) != -1:
                with_runtime = True

        if not with_runtime:
            objects += [find_lib_path("wasm_runtime.%s" % object_format)[0]]

        objects += [find_lib_path("tvmjs_support.%s" % object_format)[0]]
        objects += [find_lib_path("webgpu_runtime.%s" % object_format)[0]]

    js_output = output.replace(".wasm", ".js")

    cmd += ["-o", js_output]
    cmd += objects

    if options:
        cmd += options

    def test_copy_file(src_file, out_dir, new_filename=None):
        import shutil
        if new_filename is None:
            out_file = os.path.join(out_dir, get_filename(src_file))
        else:
            out_file = os.path.join(out_dir, new_filename)
        shutil.copyfile(src_file, out_file)
        print("emcc.py: copy %s to %s" % (src_file, out_file))
        return out_file

    def test_copy_obj_file(obj_file, out_dir, new_filename=None):
        return test_copy_file(obj_file, out_dir, new_filename)

    def test_bc_to_ll(bc_file, out_dir):
        filename = get_filename(str(bc_file))
        out_file = os.path.join(out_dir, filename.split(".")[0] + ".ll")
        test_cmd = ["llvm-dis"]
        test_cmd += ["-o", out_file]
        test_cmd += [bc_file]
        print("emcc.py: test_cmd", test_cmd)

        proc = subprocess.Popen(test_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False)
        (out, _) = proc.communicate()
        if proc.returncode != 0:
            msg = "Run llvm-dis error:\n"
            msg += py_str(out)
            raise RuntimeError(msg)

        enable_print_ll = False
        if enable_print_ll:
            cat_cmd = ["cat"]
            cat_cmd += [out_file]
            proc = subprocess.Popen(cat_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False)
            (out, _) = proc.communicate()
            print("emcc.py: LLVM IR")
            print(str(out).replace("\\n", "\n"))
            print("")
            if proc.returncode != 0:
                msg = "Run cat error:\n"
                msg += py_str(out)
                raise RuntimeError(msg)

        input("Press Enter to continue")

    def test_bc_to_obj(bc_file, out_dir):
        #print("bc_file:", bc_file)
        #print("out_dir:", out_dir)
        out_file = os.path.join(out_dir, str(bc_file).split(".")[0] + ".o")
        #print("out_file:", out_file)
        #input("Press ENTER to continue")
        
        llc_cmd = ["llc"]
        llc_cmd += ["-mtriple=wasm32-unknown-unknown-wasm"]
        llc_cmd += [opt_level]
        llc_cmd += ["-filetype=obj"]
        llc_cmd += ["-o", out_file]
        llc_cmd += [bc_file]

        print("emcc.py: llc_cmd", llc_cmd)
        #input("Press ENTER to continue")

        import time
        start_time = time.time()
        proc = subprocess.Popen(llc_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False)
        (out, _) = proc.communicate()
        print("emcc.py: llc time %.6 s" % (time.time() - start_time))

        if proc.returncode != 0:
            msg = "Run llc error:\n"
            msg += py_str(out)
            raise RuntimeError(msg)

    if enable_test:
        print("emcc.py: cmd", cmd)
        #input("Press ENTER to continue")
        if object_format == "bc":
            obj_file = objects[0]
            tmp_path = "/tmp/tvm/emcc/o3"
            #target_bc_file = test_copy_obj_file(obj_file, tmp_path)
            bc_file = obj_file
            #test_bc_to_ll(bc_file, tmp_path)
            #test_bc_to_obj(bc_file, get_dir(output))
            #exit(0)

    #proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    import sys
    if sys.platform.startswith("win"):
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    else:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=False)
    (out, _) = proc.communicate()

    if proc.returncode != 0:
        msg = "Compilation error:\n"
        msg += py_str(out)
        raise RuntimeError(msg)
    
    if os.path.exists(js_output):
        os.remove(js_output)

    def test_copy_wasm_file(wasm_file, out_dir, new_filename=None):
        return test_copy_file(wasm_file, out_dir, new_filename)

    def test_wasm_to_wat(wasm_file, out_dir):
        filename = get_filename(str(wasm_file))
        out_file = os.path.join(out_dir, filename.split(".")[0] + ".wat")

        wasm2wat_cmd = ["wasm2wat"]
        wasm2wat_cmd += [wasm_file]
        wasm2wat_cmd += ["-o", out_file]
        print("emcc.py: wasm2wat_cmd", wasm2wat_cmd)

        proc = subprocess.Popen(wasm2wat_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False)
        (out, _) = proc.communicate()
        if proc.returncode != 0:
            msg = "Run wasm2wat error:\n"
            msg += py_str(out)
            raise RuntimeError(msg)

        input("Press Enter to continue")

    def test_replace_wasm_file(wasm_file, src_wasm_file):
        filepath = get_dir(str(wasm_file))
        filename = get_filename(str(wasm_file))
        os.remove(wasm_file)
        test_copy_file(src_wasm_file, filepath, filename)

    enable_test = False
    if enable_test:
        wasm_file = output
        #tmp_path = "/tmp/tvm/emcc"
        tmp_path = "dist/wasm"
        test_copy_wasm_file(wasm_file, tmp_path, "kernel_wasm.wasm")
        #test_wasm_to_wat(wasm_file, tmp_path)
        #input("Press Enter to continue")
        #test_replace_wasm_file(wasm_file, os.path.join(tmp_path, "kernel_wasm.wasm"))


create_tvmjs_wasm.object_format = "bc"
#create_tvmjs_wasm.object_format = "ll"
#create_tvmjs_wasm.object_format = "o"
create_tvmjs_wasm.output_format = "wasm"
