#!/bin/bash

export MY_HOME=${HOME}

importTvm() {
    export TVM_HOME=$(pwd)/..
    export PYTHONPATH=${TVM_HOME}/python:${PYTHONPATH}
}

importAndActivateEmsdk() {
    # Reset your EMSDK_HOME path.
    export EMSDK_HOME=${MY_HOME}/Projects/emsdk
    ${EMSDK_HOME}/emsdk activate latest
    source ${EMSDK_HOME}/emsdk_env.sh
}

build() {
    make clean
    make
    npm run bundle
}

importTvm

importAndActivateEmsdk

build
