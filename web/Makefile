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

TVM_ROOT=$(shell cd ..; pwd)

INCLUDE_FLAGS = -I$(TVM_ROOT) -I$(TVM_ROOT)/include\
	-I$(TVM_ROOT)/3rdparty/dlpack/include -I$(TVM_ROOT)/3rdparty/dmlc-core/include -I$(TVM_ROOT)/3rdparty/compiler-rt

.PHONY: clean all rmtypedep preparetest

all: dist/wasm/tvmjs_runtime.wasm dist/wasm/tvmjs_runtime.wasi.js

EMCC = emcc

# -g
EMCC_CFLAGS = $(INCLUDE_FLAGS) -O3 -std=c++14 -Wno-ignored-attributes --no-entry \
	-sALLOW_MEMORY_GROWTH=1 -sWASM=1 -sSTANDALONE_WASM=0 -sWASM_BIGINT=1 \
	-sERROR_ON_UNDEFINED_SYMBOLS=0 -sDETERMINISTIC=0 -sASSERTIONS=1 -sUSE_PTHREADS=0 \
	-DBUILD_WASM

EMCC_LDFLAGS = --pre-js emcc/preload.js

OBJ_FORMAT = bc
#OBJ_FORMAT = o

dist/wasm/%.$(OBJ_FORMAT): emcc/%.cc
	@mkdir -p $(@D)
	$(EMCC) $(EMCC_CFLAGS) -c -MM -MT dist/wasm/$*.bc $< >dist/wasm/$*.d
	$(EMCC) $(EMCC_CFLAGS) -c -o dist/wasm/$*.$(OBJ_FORMAT) $<

dist/wasm/tvmjs_runtime.wasm: dist/wasm/wasm_runtime.$(OBJ_FORMAT) dist/wasm/tvmjs_support.$(OBJ_FORMAT) dist/wasm/webgpu_runtime.$(OBJ_FORMAT)
	@mkdir -p $(@D)
	$(EMCC) $(EMCC_CFLAGS) -o dist/wasm/tvmjs_runtime.js $+ $(EMCC_LDFLAGS)

dist/wasm/tvmjs_runtime.wasi.js: dist/wasm/tvmjs_runtime.wasm emcc/decorate_as_wasi.py
	python3 emcc/decorate_as_wasi.py dist/wasm/tvmjs_runtime.js $@

clean:
	@rm -rf dist/wasm

# Patch to remove require("@webgpu/types")
rmtypedep:
	grep -v webgpu/types dist/webgpu.js > dist/webgpu.temp.js
	mv dist/webgpu.temp.js dist/webgpu.js

-include dist/wasm/*.d
