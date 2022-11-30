import sys
import os
import pkgutil
import importlib

# List of all available op builders from deepspeed op_builder

op_builder_dir = "deepspeed.ops.op_builder"
op_builder_module = importlib.import_module(op_builder_dir)
__op_builders__ = []

this_module = sys.modules[__name__]

# reflect all builder names into variable definition such as 'TransformerBuilder = "TransformerBuilder"'
for _, module_name, _ in pkgutil.iter_modules([os.path.dirname(op_builder_module.__file__)]):
    # avoid self references
    if module_name != 'all_ops' and module_name != 'builder' and module_name != 'builder_names':
        module = importlib.import_module("{}.{}".format(op_builder_dir, module_name))
        for member_name in module.__dir__():
            if member_name.endswith(
                    'Builder'
            ) and member_name != "OpBuilder" and member_name != "CUDAOpBuilder":
                # assign builder name to variable with same name
                # the following is equivalent to i.e. TransformerBuilder = "TransformerBuilder"
                this_module.__dict__[member_name] = member_name
