import torch
import importlib

try:
    import deepspeed as ds
    print("deepspeed successfully imported")
except ImportError as err:
    raise err

print(f"torch version: {torch.__version__}")

print(f"deepspeed info: {ds.__version__}, {ds.__git_hash__}, {ds.__git_branch__}")

try:
    apex_C = importlib.import_module('apex_C')
    print("apex successfully installed")
except Exception as err:
    raise err

try:
    fused_lamb = importlib.import_module('fused_lamb_cuda')
    print('deepspeed kernels successfully installed')
except Exception as err:
    raise err

try:
    from apex.optimizers import FP16_Optimizer
    print("using old-style apex")
except ImportError:
    print("using new-style apex")
