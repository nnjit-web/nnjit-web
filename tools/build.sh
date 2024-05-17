#!/bin/bash

export CC=gcc
export CXX=g++
export BINARIEN_SDK=$(pwd)/3rdparty/binaryen

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
  make
  cd ..
}

prepareForRebuild

build
