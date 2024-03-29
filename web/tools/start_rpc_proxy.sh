#!/bin/bash

export MY_HOME=${HOME}
#export PROXY_WEB_PORT=8888
#export PROXY_PORT=9090
export PROXY_USE_SSL=1

importTvm() {
  if [ "$TVM_HOME" == "" ];then
    export TVM_HOME=${MY_HOME}/Projects/tvm-main
    #export TVM_HOME=${MY_HOME}/Projects/tvm-for-web
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

generateSSLCertificate() {
  openssl req -newkey rsa:4096 -x509 -sha256 -days 3650 -nodes \
      -out ssl/certificate.crt -keyout ssl/privatekey.key
  # Country Name: CN
  # State or Province Name: Beijing
  # Locality Name: Beijing
  # Organization Name: Microsoft
  # Organizational Unit Name: HEX
  # Common Name: Fucheng Jia
  # Email Address: v-fuchengjia@microsoft.com
}

startRpcProxy() {
  python -m tvm.exec.rpc_proxy --example-rpc=1 \
      --web-port=${PROXY_WEB_PORT} --port=${PROXY_PORT} --timeout=86400
}

startRpcProxyWithSSL() {
  export SSL_HOME=$(pwd)/ssl
  python -m tvm.exec.rpc_proxy --example-rpc=1 \
      --web-port=${PROXY_WEB_PORT} --port=${PROXY_PORT} --timeout=86400 \
      --certfile=${SSL_HOME}/certificate.crt \
      --keyfile=${SSL_HOME}/privatekey.key
}

importTvm

checkArgNotNone "PROXY_WEB_PORT" $PROXY_WEB_PORT
checkArgNotNone "PROXY_PORT" $PROXY_PORT

#generateSSLCertificate

if [ "$PROXY_USE_SSL" == "1" ];then
  startRpcProxyWithSSL
else
  startRpcProxy
fi
