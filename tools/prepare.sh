#!/bin/bash

installRequirements() {
  # CMake
  conda install cmake
  #sudo apt install cmake
  # NPM
  #sudo apt install npm
  # GCC and GXX
  conda install -c conda-forge gcc_linux-64=8.4.0 gxx_linux-64=8.4.0
  # CLANG and LLD
  conda install -c conda-forge clang=11.0 lld=11.0
  # LLVM 11.0
  conda install -c conda-forge libllvm11=11.0 lit=11.0 llvm=11.0 llvm-tools=11.0 llvmdev=11.0
  conda install -c conda-forge libllvm14=14.0 lit=14.0 llvm=14.0 llvm-tools=14.0 llvmdev=14.0
  #conda uninstall libllvm11 lit llvm llvm-tools llvmdev
  # Vulkan SDK
  # wget https://...
  # Emscripten
  # ...
}

initGitSubmodules() {
  git submodule init
  git submodule update
}

#installRequirements

initGitSubmodules
