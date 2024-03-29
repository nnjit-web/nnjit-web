
## 0. Requirements

* Anaconda (Python 3.7 is recommended)
* CMake (3.22.1 is recommended)
* LLVM (11.0 is recommended)
```shell
conda install -c conda-forge libllvm11=11.0 lit=11.0 llvm=11.0 llvm-tools=11.0 llvmdev=11.0
```
* VulkanSDK (1.3.224.1 is recommended)
  * Download at https://vulkan.lunarg.com/
* Emscripten
  * Refer to https://emscripten.org/docs/getting_started/downloads.html
* Chromium (111.0.5555.0 64-bit is recommended)
  * For Windows, download at https://www.googleapis.com/download/storage/v1/b/chromium-browser-snapshots/o/Win_x64%2F1095473%2Fmini_installer.exe?generation=1674417684350627&alt=media)

### On Windows

* Clang and LLD (11.0 is recommended)
```
conda install -c conda-forge clang=11.0 lld=11.0
```

### On Linux

* GCC (7.5.0 is recommended)

## 1. Build library

### Step 1-1: Modify ./cmake/config.cmake

```shell
#set(USE_VULKAN OFF)
# Enable Vulkan for WebGPU kernel generation.
# On Windows, e.g.,
set(USE_VULKAN D:/VulkanSDK/1.3.224.1)

# Enable Binaryen.
set(USE_BINARYEN ON)
```

### Step 1-2: Git Initilization

```shell
git submodule init
git submodule update
```

### Step 1-3: Build

On Windows:

```shell
call .\tools\build.bat
```

On Linux:

```shell
bash ./tools/build.sh
```

## 2. NNJIT kernel optimization

## Step 2-0: Install requirements for nnjit-web

On Windows:

```shell
cd .\web
call tools\install_requirements.bat
```

On Linux:

```shell
cd ./web
bash tools/install_requirements.sh
```

## Step 2-1: Start RPC proxy for browser connection

In `web` directory,

On Windows:

```shell
call tools\start_rpc_proxy.bat
```

On Linux:

```shell
PROXY_WEB_PORT=8888 PROXY_PORT=9090 bash tools/start_rpc_proxy.sh
```

NOTE: You should keep this terminal open until all below tests finish.

## Step 2-2: Open Chrominum browser and connect to RPC proxy

On Windows:

```shell
set CHROME_PATH=C:\Users\%username%\AppData\Local\Chromium\Application\chrome.exe

"%CHROME_PATH%" https://127.0.0.1:8888/ --enable-unsafe-webgpu --disable-dawn-features=disallow_unsafe_apis
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

In `web` directory, you should (1) modify the `EMSDK_HOME` in the file `tools\test_nnjit_for_models.bat` according to your path, and (2) change the device name "dell-g5-5090" below accoding to the RPC server key in your browser.

On Windows:

```shell
call tools\test_nnjit_for_models.bat dell-g5-5090 [roberta | bart | gpt-2 | t5-encoder] [wasm | webgpu] [tvm | nnjit]
```

On Linux:

```shell
bash tools/test_nnjit_for_models.sh dell-g5-5090 [roberta | bart | gpt-2 | t5-encoder] [wasm | webgpu] [tvm | nnjit]
```
