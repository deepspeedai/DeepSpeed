# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os

try:
    # Importing logger currently requires that torch is installed, hence the try...except
    # TODO: Remove logger dependency on torch.
    from deepspeed.utils import logger as accel_logger
except ImportError as e:
    accel_logger = None

try:
    from accelerator.abstract_accelerator import DeepSpeedAccelerator as dsa1
except ImportError as e:
    dsa1 = None
try:
    from deepspeed.accelerator.abstract_accelerator import DeepSpeedAccelerator as dsa2
except ImportError as e:
    dsa2 = None

ds_accelerator = None


def _validate_accelerator(accel_obj):
    # because abstract_accelerator has different path during
    # build time (accelerator.abstract_accelerator)
    # and run time (deepspeed.accelerator.abstract_accelerator)
    # and extension would import the
    # run time abstract_accelerator/DeepSpeedAccelerator as its base
    # class, so we need to compare accel_obj with both base class.
    # if accel_obj is instance of DeepSpeedAccelerator in one of
    # accelerator.abstractor_accelerator
    # or deepspeed.accelerator.abstract_accelerator, consider accel_obj
    # is a conforming object
    if not ((dsa1 != None and isinstance(accel_obj, dsa1)) or (dsa2 != None and isinstance(accel_obj, dsa2))):
        raise AssertionError(f'{accel_obj.__class__.__name__} accelerator is not subclass of DeepSpeedAccelerator')

    # TODO: turn off is_available test since this breaks tests
    #assert accel_obj.is_available(), \
    #    f'{accel_obj.__class__.__name__} accelerator fails is_available() test'


def get_accelerator():
    global ds_accelerator
    if ds_accelerator is not None:
        return ds_accelerator

    accelerator_name = None
    ds_set_method = None
    # 1. Detect whether there is override of DeepSpeed accelerators from environment variable.
    #    DS_ACCELERATOR = 'cuda'|'xpu'|'cpu'
    if 'DS_ACCELERATOR' in os.environ.keys():
        accelerator_name = os.environ['DS_ACCELERATOR']
        if accelerator_name == 'xpu':
            try:
                from intel_extension_for_deepspeed import XPU_Accelerator  # noqa: F401
            except ImportError as e:
                raise ValueError(
                    f'XPU_Accelerator requires intel_extension_for_deepspeed, which is not installed on this system.')
        elif accelerator_name == 'cpu':
            try:
                import intel_extension_for_pytorch  # noqa: F401
            except ImportError as e:
                raise ValueError(
                    f'CPU_Accelerator requires intel_extension_for_pytorch, which is not installed on this system.')
        elif accelerator_name == 'cuda':
            pass
        else:
            raise ValueError(
                f'DS_ACCELERATOR must be one of "cuda", "cpu", or "xpu".  Value "{accelerator_name}" is not supported')
        ds_set_method = 'override'

    # 2. If no override, detect which accelerator to use automatically
    if accelerator_name == None:
        try:
            from intel_extension_for_deepspeed import XPU_Accelerator  # noqa: F401,F811
            accelerator_name = 'xpu'
        except ImportError as e:
            # We need a way to choose between CUDA_Accelerator and CPU_Accelerator
            # Currently we detect whether intel_extension_for_pytorch is installed
            # in the environment and use CPU_Accelerator if the answer is True.
            # An alternative might be detect whether CUDA device is installed on
            # the system but this comes with two pitfalls:
            # 1. the system may not have torch pre-installed, so
            #    get_accelerator().is_available() may not work.
            # 2. Some scenario like install on login node (without CUDA device)
            #    and run on compute node (with CUDA device) may cause mismatch
            #    between installation time and runtime.
            try:
                import intel_extension_for_pytorch  # noqa: F401,F811
                accelerator_name = 'cpu'
            except ImportError as e:
                accelerator_name = 'cuda'
        ds_set_method = 'auto detect'

    # 3. Set ds_accelerator accordingly
    if accelerator_name == 'cuda':
        from .cuda_accelerator import CUDA_Accelerator
        ds_accelerator = CUDA_Accelerator()
    elif accelerator_name == 'cpu':
        from .cpu_accelerator import CPU_Accelerator
        ds_accelerator = CPU_Accelerator()
    elif accelerator_name == 'xpu':
        # XPU_Accelerator is already imported in detection stage
        ds_accelerator = XPU_Accelerator()
    _validate_accelerator(ds_accelerator)
    if accel_logger is not None:
        accel_logger.info(f"Setting ds_accelerator to {ds_accelerator._name} ({ds_set_method})")
    return ds_accelerator


def set_accelerator(accel_obj):
    global ds_accelerator
    _validate_accelerator(accel_obj)
    if accel_logger is not None:
        accel_logger.info(f"Setting ds_accelerator to {accel_obj._name} (model specified)")
    ds_accelerator = accel_obj


'''
-----------[code] test_get.py -----------
from deepspeed.accelerator import get_accelerator
my_accelerator = get_accelerator()
logger.info(f'{my_accelerator._name=}')
logger.info(f'{my_accelerator._communication_backend=}')
logger.info(f'{my_accelerator.HalfTensor().device=}')
logger.info(f'{my_accelerator.total_memory()=}')
-----------[code] test_get.py -----------

---[output] python test_get.py---------
my_accelerator.name()='cuda'
my_accelerator.communication_backend='nccl'
my_accelerator.HalfTensor().device=device(type='cuda', index=0)
my_accelerator.total_memory()=34089730048
---[output] python test_get.py---------

**************************************************************************
-----------[code] test_set.py -----------
from deepspeed.accelerator.cuda_accelerator import CUDA_Accelerator
cu_accel = CUDA_Accelerator()
logger.info(f'{id(cu_accel)=}')
from deepspeed.accelerator import set_accelerator, get_accelerator
set_accelerator(cu_accel)

my_accelerator = get_accelerator()
logger.info(f'{id(my_accelerator)=}')
logger.info(f'{my_accelerator._name=}')
logger.info(f'{my_accelerator._communication_backend=}')
logger.info(f'{my_accelerator.HalfTensor().device=}')
logger.info(f'{my_accelerator.total_memory()=}')
-----------[code] test_set.py -----------


---[output] python test_set.py---------
id(cu_accel)=139648165478304
my_accelerator=<deepspeed.accelerator.cuda_accelerator.CUDA_Accelerator object at 0x7f025f4bffa0>
my_accelerator.name='cuda'
my_accelerator.communication_backend='nccl'
my_accelerator.HalfTensor().device=device(type='cuda', index=0)
my_accelerator.total_memory()=34089730048
---[output] python test_set.py---------
'''
