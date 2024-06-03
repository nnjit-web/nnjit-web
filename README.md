
## 0. Requirements

* Ubuntu (>=18.04 is recommended)
* Anaconda (Python >=3.7 is recommended)
* CMake (>=3.22.1 is recommended)
* LLVM (11.0 is recommended)
```shell
conda install -c conda-forge libllvm11=11.0 lit=11.0 llvm=11.0 llvm-tools=11.0 llvmdev=11.0
```
* GCC (>=7.5.0 is recommended)

## 1. Build library

### Step 1-1: Build submodules

```shell
git submodule init
git submodule update
bash ./tools/build_binaryen.sh
bash ./tools/build_emsdk.sh
```

### Step 1-2: Build TVM

```shell
bash ./tools/build.sh
```

## 2. NNJIT kernel optimization

## Step 2-0: Install and build requirements for nnjit-web

```shell
cd ./web
bash ./tools/install_requirements.sh
bash ./tools/build.sh
```

## Step 2-1: Start RPC proxy for browser connection

In `web` directory,

```shell
bash tools/start_rpc_proxy.sh
```

NOTE: You should keep this terminal open until all below tests finish.

## Step 2-2: Open Chrominum browser and connect to RPC proxy

```shell
set SERVER_IP=127.0.0.1
set CHROME_PATH=C:\Users\%username%\AppData\Local\Chromium\Application\chrome.exe

"%CHROME_PATH%" https://%SERVER_IP%:8888/ --enable-unsafe-webgpu --enable-dawn-features=allow_unsafe_apis
```

## Step 2-3: Set RPC server key in browser according to your device name

In the browser page, the default RPC server key is "wasm". It is recommended to set this key according to your device name such as "dell-g5-5090" and "hornor-magicbook-16". This key is used as a part of final JSON file name, so that we can identify which device the file belongs to.

Then click the "Connect to Proxy" button, you will see the successful log as below:

```shell
WebSocketRPCServer[dell-g5-5090]: connected...
```

## Step 2-4: Download ONNX models

Download four examples ONNX models from https://1drv.ms/f/s!AsM4OHFFcOSAtWHPE4DRj1QgrBTh?e=rECKZl.

Then put all .onnx files to `./onnx_models` directory as follows. If the directory does not exist, you should create it by yourself.

```shell
nnjit-web/
  web/
    onnx_models/
      bart.onnx
      gpt2-10.onnx
      roberta.onnx
      t5-small-encoder.onnx
```

## Step 2-5: Run a script to JIT optimize kernel for models

In `web` directory, you should change the device name "dell-g5-5090" below accoding to the RPC server key in your browser.


```shell
bash ./tools/test_nnjit_for_models.sh dell-g5-5090 [roberta | bart | gpt-2 | t5-encoder] [wasm | webgpu] [tvm | nnjit]
```

Kernel configs are saved in `logs` directory.
