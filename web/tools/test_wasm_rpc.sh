#!/bin/bash

export MY_HOME=${HOME}
export PROXY_PORT=9090
export DEV_INFO=$1
export MODEL_NAME=$2
export BACKEND=$3
export ENGINE=$4

importTvm() {
    if [ "$TVM_HOME" == "" ];then
        export TVM_HOME=$(pwd)/..
    fi
    export PYTHONPATH=${TVM_HOME}/python:${PYTHONPATH}
}

importAndActivateEmsdk() {
    export EMSDK_HOME=$(pwd)/../3rdparty/emsdk
    ${EMSDK_HOME}/emsdk activate latest
    source ${EMSDK_HOME}/emsdk_env.sh
}

wasmToWat() {
    export BINARYEN_BIN_PATH=${TVM_HOME}/3rdparty/binaryen/build/bin
    export PATH=${BINARYEN_BIN_PATH}:${PATH}
    wasm-dis temp/addone.wasm -o temp/addone.wat
}

importTvm

importAndActivateEmsdk

export TVM_LOG_DEBUG="DEFAULT=2"

python tests/python/websock_rpc_test.py

wasmToWat
