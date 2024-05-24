#!/bin/bash

export MY_HOME=${HOME}
export PROXY_USE_SSL=1
export PROXY_WEB_PORT=8888
export PROXY_PORT=9090

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

startRpcProxy() {
    python -m tvm.exec.rpc_proxy --example-rpc=1 \
        --web-port=${PROXY_WEB_PORT} --port=${PROXY_PORT}
}

startRpcProxyWithSSL() {
    export SSL_HOME=$(pwd)/ssl
    python -m tvm.exec.rpc_proxy --example-rpc=1 \
        --web-port=${PROXY_WEB_PORT} --port=${PROXY_PORT} \
        --certfile=${SSL_HOME}/certificate.crt \
        --keyfile=${SSL_HOME}/privatekey.key
}

importTvm

checkArgNotNone "PROXY_WEB_PORT" $PROXY_WEB_PORT
checkArgNotNone "PROXY_PORT" $PROXY_PORT

if [ "$PROXY_USE_SSL" == "1" ];then
    startRpcProxyWithSSL
else
    startRpcProxy
fi
