import torch
import warnings
import importlib
import warnings

GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
END = '\033[0m'
SUCCESS = f"{GREEN} [SUCCESS] {END}"
WARNING = f"{YELLOW} [WARNING] {END}"
FAIL = f'{RED} [FAIL] {END}'
INFO = ' [INFO]'

try:
    import deepspeed
    print(f"{SUCCESS} deepspeed successfully imported.")
except ImportError as err:
    raise err

print(f"{INFO} torch install path: {torch.__path__}")
print(f"{INFO} torch version: {torch.__version__}, torch.cuda: {torch.version.cuda}")
print(f"{INFO} deepspeed install path: {deepspeed.__path__}")
print(
    f"{INFO} deepspeed info: {deepspeed.__version__}, {deepspeed.__git_hash__}, {deepspeed.__git_branch__}"
)

try:
    apex_C = importlib.import_module('apex_C')
    print(f"{SUCCESS} apex extensions successfully installed")
except Exception as err:
    print(f'{WARNING} apex extensions are not installed')

try:
    from apex.optimizers import FP16_Optimizer
    print(f"{INFO} using old-style apex")
except ImportError:
    print(f"{INFO} using new-style apex")

try:
    importlib.import_module('deepspeed.ops.lamb.fused_lamb_cuda')
    print(f'{SUCCESS} fused lamb successfully installed.')
except Exception as err:
    print(f"{WARNING} fused lamb is NOT installed.")

try:
    importlib.import_module('deepspeed.ops.transformer.transformer_cuda')
    print(f'{SUCCESS} transformer kernels successfully installed.')
except Exception as err:
    print(f'{WARNING} transformer kernels are NOT installed.')

try:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        importlib.import_module('deepspeed.ops.sparse_attention.cpp_utils')
        import triton
    print(f'{SUCCESS} sparse attention successfully installed.')
except ImportError:
    print(f'{WARNING} sparse attention is NOT installed.')

try:
    importlib.import_module('deepspeed.ops.adam.cpu_adam_op')
    print(f'{SUCCESS} cpu-adam (used by ZeRO-offload) successfully installed.')
except ImportError:
    print(f'{WARNING} cpu-adam (used by ZeRO-offload) is NOT installed.')
