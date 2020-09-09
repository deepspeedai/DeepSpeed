import torch
import warnings
import importlib

try:
    import deepspeed
    print("deepspeed successfully imported")
except ImportError as err:
    raise err

print(f"torch install path: {torch.__path__}")
print(f"torch version: {torch.__version__}")
print(f"deepspeed install path: {deepspeed.__path__}")
print(
    f"deepspeed info: {deepspeed.__version__}, {deepspeed.__git_hash__}, {deepspeed.__git_branch__}"
)

try:
    apex_C = importlib.import_module('apex_C')
    print("apex successfully installed")
except Exception as err:
    raise err

try:
    from apex.optimizers import FP16_Optimizer
    print("using old-style apex")
except ImportError:
    print("using new-style apex")

try:
    importlib.import_module('deepspeed.ops.lamb.fused_lamb_cuda')
    print('deepspeed lamb successfully installed.')
except Exception as err:
    warnings.warn("deepspeed lamb is NOT installed.")

try:
    importlib.import_module('deepspeed.ops.transformer.transformer_cuda')
    print('deepspeed transformer kernels successfully installed.')
except Exception as err:
    warnings.warn('deepspeed transformer kernels are NOT installed.')

try:
    importlib.import_module('deepspeed.ops.sparse_attention.cpp_utils')
    print('deepspeed sparse attention successfully installed.')
except ImportError:
    warnings.warn('deepspeed sparse attention is NOT installed.')
