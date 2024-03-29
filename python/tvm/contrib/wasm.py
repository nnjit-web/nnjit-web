
import os
import shutil
from .utils import get_os_env_var_bool
from .utils import get_dir, get_filename


def create_wasm(output, objects, options=None, cc=None):
    if get_os_env_var_bool("TVM_ENABLE_BUILD_LOG", False):
        print("wasm.py: objects:", objects)
        print("wasm.py: output:", output)
    objects = [objects] if isinstance(objects, str) else objects
    shutil.copyfile(objects[0], output)

    enable_debug = False
    if enable_debug:
        debug_output = "dist/wasm/kernel_wasm.wasm"
        shutil.copyfile(objects[0], debug_output)
        print("wasm.py: copy_src %s, copy_dst %s" % (objects[0], debug_output))

    enable_modified_wasm = False
    if enable_debug and enable_modified_wasm:
        wasm_input = "dist/wasm/kernel_wasm_fast.wasm"
        wasm_output = output
        if os.path.exists(wasm_output):
            os.remove(wasm_output)
        shutil.copyfile(wasm_input, wasm_output)
        print("wasm.py: copy_src %s, copy_dst %s" % (wasm_input, wasm_output))

    #input("wasm.py: Press ENTER to continue")


create_wasm.object_format = "wasm"
create_wasm.output_format = create_wasm.object_format
