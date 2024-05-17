#!/bin/bash

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
  make -j12
  cd ..
}

prepareForRebuild

build
