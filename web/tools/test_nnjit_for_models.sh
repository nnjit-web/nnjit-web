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

checkArgNotNone() {
    ARG_NAME=$1
    ARG_VALUE=$2
    if [ "$ARG_VALUE" == "" ];then
        echo "Please set ${ARG_NAME}"
        exit
    fi
}

importAndActivateEmsdk() {
    export EMSDK_HOME=$(pwd)/../3rdparty/emsdk
    ${EMSDK_HOME}/emsdk activate latest
    source ${EMSDK_HOME}/emsdk_env.sh
}

clearEmsdkSymbolLists() {
    if [ "$EMSDK_HOME" == "" ];then
        export EMSDK_HOME=${MY_HOME}/Software/emsdk
    fi
    rm -rf ${EMSDK_HOME}/upstream/emscripten/cache/symbol_lists/*.json
}

clearEmsdkCache() {
    emcc --clear-cache
}

runAutoTvmOneModelTest() {
    export MODEL_NAME=$1

    export TVM_ENABLE_TIR_LOG=0
    #export TVM_LOG_DEBUG=1
    #export TVM_LOG_DEBUG=DEFAULT=0
    #export TVM_LOG_DEBUG="DEFAULT=1,src/relay/ir/dataflow_matcher.cc=-1,src/relay/ir/transform.cc=-1,src/relay/ir/indexed_graph.cc=-1,src/ir/transform.cc=-1"
    #export TVM_LOG_DEBUG="DEFAULT=-1,src/relay/backend/build_module.cc=1"
    export TVM_ENABLE_GEMM_LOG=1
    export TVM_ENABLE_TUNING=1
    export EMCC_OPT_LEVEL=-O3
    export EMCC_USE_SIMD=1
    export TVM_TUNING_SPACE_NAME=default
    export TVM_ENABLE_HW_VERYFICATION=0

    if [ "${BACKEND}" == "" ] | [ "${BACKEND}" == "all" ];then
        python tests/python/model_tuning_autotvm_test.py \
            --model-name=${MODEL_NAME} --dev-info=${DEV_INFO} --backend=llvm-wasm
        python tests/python/model_tuning_autotvm_test.py \
            --model-name=${MODEL_NAME} --dev-info=${DEV_INFO} --backend=tvm-webgpu
    else
        if [ "${BACKEND}" == "wasm" ];then
            export BACKEND=llvm-wasm
        fi
        if [ "${BACKEND}" == "webgpu" ];then
            export BACKEND=tvm-webgpu
        fi
        python tests/python/model_tuning_autotvm_test.py \
            --model-name=${MODEL_NAME} --dev-info=${DEV_INFO} --backend=${BACKEND}
    fi
}

runAutoTvmAllModelsTest() {
    runAutoTvmOneModelTest roberta
    runAutoTvmOneModelTest bart
    runAutoTvmOneModelTest gpt-2
    runAutoTvmOneModelTest t5-encoder
    #runAutoTvmOneModelTest vgg-16
}

runOnlineGenOneModelTest() {
    export MODEL_NAME=$1

    export TVM_ENABLE_TIR_LOG=0
    #export TVM_LOG_DEBUG=1
    #export TVM_LOG_DEBUG="DEFAULT=1"
    #export TVM_LOG_DEBUG="DEFAULT=-1,src/ir/transform.cc=-1,src/relay/backend/te_compiler.cc=1"
    #export TVM_BACKTRACE=1
    export TVM_ENABLE_TUNING=1
    export EMCC_OPT_LEVEL=-O3
    export EMCC_USE_SIMD=1
    export TVM_TUNING_SPACE_NAME=small
    export TVM_ENABLE_HW_VERYFICATION=1
    export TVM_ENABLE_SAVE_ERROR_RESULT=0
    export TVM_ENABLE_VERIFICATION_LOG=1

    if [ "${BACKEND}" == "" ] | [ "${BACKEND}" == "all" ];then
        python tests/python/model_tuning_autotvm_test.py \
            --model-name=${MODEL_NAME} --dev-info=${DEV_INFO} --backend=wasm
        python tests/python/model_tuning_autotvm_test.py \
            --model-name=${MODEL_NAME} --dev-info=${DEV_INFO} --backend=webgpu
    else
        #if [ "${BACKEND}" == "wasm" ];then
        #    export BACKEND=llvm-wasm
        #fi
        python tests/python/model_tuning_autotvm_test.py \
            --model-name=${MODEL_NAME} --dev-info=${DEV_INFO} --backend=${BACKEND}
    fi
}

runOnlineGenAllModelsTest() {
    runOnlineGenOneModelTest roberta
    runOnlineGenOneModelTest bart
    runOnlineGenOneModelTest gpt-2
    runOnlineGenOneModelTest t5-encoder
    #runOnlineGenOneModelTest vgg-16
}

runTest() {
    if [ "${MODEL_NAME}" == "" ] | [ "${MODEL_NAME}" == "all" ];then
        if [ "${ENGINE}" == "tvm" ];then
            runAutoTvmAllModelsTest
        fi
        if [ "${ENGINE}" == "nnjit" ];then
            runOnlineGenAllModelsTest
        fi
    else
        if [ "${ENGINE}" == "tvm" ];then
            runAutoTvmOneModelTest "${MODEL_NAME}"
        fi
        if [ "${ENGINE}" == "nnjit" ];then
            runOnlineGenOneModelTest "${MODEL_NAME}"
        fi
    fi
}

importTvm

checkArgNotNone "PROXY_PORT" $PROXY_PORT
checkArgNotNone "DEV_INFO" $DEV_INFO

importAndActivateEmsdk

#clearEmsdkSymbolLists

#clearEmsdkCache

runTest
