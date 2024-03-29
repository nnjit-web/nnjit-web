
import subprocess

proxy_key = "honor-magicbook-16"
model_name = "roberta"
backend_name = "llvm-wasm"
engine_name = "tvm"

cmd = ["bash", "tools/test_model_tuning.sh", proxy_key, model_name, backend_name, engine_name]

log_filepath = "logs/%s-%s-%s-%s.txt" % (proxy_key, model_name, backend_name, engine_name)
log_file = open(log_filepath, "w")

p = subprocess.Popen(cmd, shell=False, stdout=log_file, stderr=subprocess.STDOUT)

stdout, stderr = p.communicate()

print(str(stdout, encoding="utf-8"))
