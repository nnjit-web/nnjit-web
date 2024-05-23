#!/bin/bash

installNodeJS() {
    sudo apt install nodejs
    sudo apt install npm
    sudo npm cache clean -f
    sudo npm install -g n
    sudo n 16.18.0
}

installNpmPackages() {
    npm i
    npm install typescript
    npm i @types/node
    npm install @webgpu/types
}

installPipPackages() {
    pip install decorator
    pip install psutil
    pip install scipy
    pip install attrs

    pip install tornado
    pip install cloudpickle

    pip install onnx
    pip install pillow
    pip install xgboost==1.5.2

    pip install pytest
    pip install typing-extensions
}

installNodeJS

installNpmPackages

installPipPackages
