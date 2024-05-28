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

importTvm

importAndActivateEmsdk

python tests/python/webgpu_rpc_test.py
