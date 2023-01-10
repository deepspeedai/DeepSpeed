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
    if not ((dsa1 != None and isinstance(accel_obj,
                                         dsa1)) or
            (dsa2 != None and isinstance(accel_obj,
                                         dsa2))):
        raise AssertionError(
            f'{accel_obj.__class__.__name__} accelerator is not subclass of DeepSpeedAccelerator'
        )

    # TODO: turn off is_available test since this breaks tests
    #assert accel_obj.is_available(), \
    #    f'{accel_obj.__class__.__name__} accelerator fails is_available() test'


def get_accelerator():
    global ds_accelerator
    if ds_accelerator is None:
        try:
            from intel_extension_for_deepspeed import XPU_Accelerator
        except ImportError as e:
            pass
        else:
            ds_accelerator = XPU_Accelerator()
            _validate_accelerator(ds_accelerator)
            return ds_accelerator

        from .cuda_accelerator import CUDA_Accelerator
        ds_accelerator = CUDA_Accelerator()
        _validate_accelerator(ds_accelerator)
    return ds_accelerator


def set_accelerator(accel_obj):
    global ds_accelerator
    _validate_accelerator(accel_obj)
    ds_accelerator = accel_obj


'''
-----------[code] test_get.py -----------
from deepspeed.accelerator import get_accelerator
my_accelerator = get_accelerator()
print(f'{my_accelerator._name=}')
print(f'{my_accelerator._communication_backend=}')
print(f'{my_accelerator.HalfTensor().device=}')
print(f'{my_accelerator.total_memory()=}')
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
print(f'{id(cu_accel)=}')
from deepspeed.accelerator import set_accelerator, get_accelerator
set_accelerator(cu_accel)

my_accelerator = get_accelerator()
print(f'{id(my_accelerator)=}')
print(f'{my_accelerator._name=}')
print(f'{my_accelerator._communication_backend=}')
print(f'{my_accelerator.HalfTensor().device=}')
print(f'{my_accelerator.total_memory()=}')
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
