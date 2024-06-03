
import os


def set_proxy_key_in_os_envs(dev_info):
    if dev_info == "unknown":
        key = "wasm"
    else:
        key = dev_info
    os.environ["PROXY_KEY"] = key


def set_target_name_in_os_envs(target_name):
    os.environ["TARGET_NAME"] = target_name


def get_os_env_var_bool(var_name, default_var):
    import os
    var = os.getenv(var_name)
    if var is None:
        return default_var
    return True if var == "1" else False


def get_os_env_var_int(var_name, default_var):
    import os
    var = os.getenv(var_name)
    if var is None:
        return int(default_var)
    return int(var)
