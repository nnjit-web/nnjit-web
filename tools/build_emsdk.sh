#!/bin/bash

build() {
    git pull
    ./emsdk install latest
}

cd 3rdparty/emsdk

build
