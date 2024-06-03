#!/bin/bash

export CC=gcc-7
export CXX=g++-7
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
    make -j12
    cd ..
}

prepareForRebuild

build
