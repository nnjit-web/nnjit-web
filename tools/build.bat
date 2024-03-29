
set CLANG_PATH=clang
::set CLANG_PATH="C:/Users/v-fuchengjia/.conda/envs/py37/Library/bin/clang.exe"
::set CLANG_PATH="D:/Program Files (x86)/Microsoft Visual Studio/2019/Enterprise/VC/Tools/Llvm/x64/bin/clang.exe"

set CC=%CLANG_PATH%
set CXX=%CLANG_PATH%
::set LD=lld
::set LLD=lld

if not exist "build" mkdir build

del /s /q build\*

xcopy /y cmake\config.cmake build

cd build
cmake -A x64 -Thost=x64 -D CMAKE_C_COMPILER=%CC% -D CMAKE_CXX_COMPILER=%CXX% ..
cd ..

cmake --build build --config Release -- /m
