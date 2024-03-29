
set PROXY_WEB_PORT=8888
set PROXY_PORT=9090

set TVM_HOME=%cd%\..
set PYTHONPATH=%TVM_HOME%\python;%PYTHONPATH%
set SSL_HOME=%TVM_HOME%\web\ssl

python -m tvm.exec.rpc_proxy --example-rpc=1 ^
    --web-port=%PROXY_WEB_PORT% --port=%PROXY_PORT% ^
    --certfile=%SSL_HOME%\certificate.crt ^
    --keyfile=%SSL_HOME%\privatekey.key
