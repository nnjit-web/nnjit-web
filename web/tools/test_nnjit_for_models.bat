
SET DEV_INFO=%1

SET EMSDK_HOME=C:\ProgramFiles\emsdk
CALL %EMSDK_HOME%\emsdk activate latest

SET TVM_HOME=%cd%\..
SET PYTHONPATH=%TVM_HOME%\python;%PYTHONPATH%

CALL :TUNE_ALL_MODELS
GOTO :END

:TUNE_ONE_MODEL
SET TVM_ENABLE_PRINT_TIR=0
SET TVM_LOG_DEBUG=0
SET TVM_ENABLE_TUNING=1
SET EMCC_OPT_LEVEL=-O3
SET EMCC_USE_SIMD=1

python tests\python\model_tuning_autotvm_test.py ^
  --model-name=%MODEL_NAME% --dev-info=%DEV_INFO% --backend=llvm-wasm
python tests\python\model_tuning_autotvm_test.py ^
  --model-name=%MODEL_NAME% --dev-info=%DEV_INFO% --backend=webgpu
EXIT /B 0

:TUNE_ALL_MODELS
SET MODEL_NAME=roberta
CALL :TUNE_ONE_MODEL
SET MODEL_NAME=bart
CALL :TUNE_ONE_MODEL
SET MODEL_NAME=gpt-2
CALL :TUNE_ONE_MODEL
SET MODEL_NAME=t5-encoder
CALL :TUNE_ONE_MODEL
EXIT /B 0

:END
