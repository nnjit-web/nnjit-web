#!/bin/bash

export MY_HOME=${HOME}

importTvm() {
  export PROJECT_HOME=${MY_HOME}/Projects
  export TVM_HOME=${PROJECT_HOME}/tvm-main
  export PYTHONPATH=${TVM_HOME}/python:${PYTHONPATH}
}

importAndActivateEmsdk() {
  export EMSDK_HOME=${MY_HOME}/Software/emsdk
  ${EMSDK_HOME}/emsdk activate latest
  source ${EMSDK_HOME}/emsdk_env.sh
}

build() {
  make clean
  make -j16
  npm run bundle
}

importTvm

importAndActivateEmsdk

#bash tools/install_requirements.sh

build
