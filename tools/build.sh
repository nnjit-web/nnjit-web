#!/bin/bash

export MY_HOME=${HOME}
export CONDA_ENV_HOME=${MY_HOME}/Software/anaconda3/envs/py37
#export CC=${CONDA_ENV_HOME}/bin/clang
#export CXX=${CONDA_ENV_HOME}/bin/clang
#export LD=${CONDA_ENV_HOME}/bin/lld
#export LLD=${CONDA_ENV_HOME}/bin/lld
export CC=gcc
export CXX=g++

prepare() {
  mkdir -p build
}

cleanFiles() {
  rm -rf build/*
}

copyFiles() {
  cp cmake/config.cmake build
}

prepareForRebuild() {
  prepare
  #cleanFiles
  copyFiles
}

build() {
  cd build
  cmake ..
  make -j2
  cd ..
}

#prepareForRebuild

build
