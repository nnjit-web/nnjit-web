
import shutil


def create_wasm(output, objects, options=None, cc=None):
    objects = [objects] if isinstance(objects, str) else objects
    shutil.copyfile(objects[0], output)


create_wasm.object_format = "wasm"

create_wasm.output_format = create_wasm.object_format
