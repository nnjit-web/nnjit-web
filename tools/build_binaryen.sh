#!/bin/bash

export CC=gcc
export CXX=g++

prepare() {
    git submodule init
    git submodule update
}

cleanFiles() {
    rm -rf build/*
}

build() {
    mkdir -p build
    cd build
    cmake ..
    make -j12
}

cd 3rdparty/binaryen

prepare

build
