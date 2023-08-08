# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

#!/usr/bin/env python

# run the benchmark under timeit (-t), cProfile (-c), line_profiler (-l)
#
# usage:
# ./unflatten_bench.py -t
# ./unflatten_bench.py -c
# kernprof -l unflatten_bench.py -l; python -m line_profiler  unflatten_bench.py.lprof

import argparse
import gc
import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import UtilsBuilder

from apex_C import flatten as flatten_apex
from apex_C import unflatten as unflatten_apex

util_ops = UtilsBuilder().load()
flatten = util_ops.flatten
unflatten = util_ops.unflatten

torch.manual_seed(0)
# emulate a small typical model weights
x = [
    torch.rand((512, 512)).to(get_accelerator().device_name()),
    torch.rand((512, 1024)).to(get_accelerator().device_name()),
    torch.rand((512, 30000)).to(get_accelerator().device_name())
]
unflat_t = x * 30

# warm up and check that the same output is produced
flat_py = _flatten_dense_tensors(unflat_t)
flat_cpp = flatten(unflat_t)
flat_apex = flatten_apex(unflat_t)
#numel = flat_cpp.numel()
assert torch.eq(flat_py, flat_cpp).all(), "both produce the same tensor"
assert torch.eq(flat_py, flat_apex).all(), "both produce the same tensor"

flat_t = flat_py
unflat_py = _unflatten_dense_tensors(flat_py, unflat_t)
for i in range(len(unflat_t)):
    assert torch.eq(unflat_t[i], unflat_py[i]).all()
unflat_cpp = _unflatten_dense_tensors(flat_cpp, unflat_t)
for i in range(len(unflat_t)):
    assert torch.eq(unflat_t[i], unflat_cpp[i]).all()
unflat_apex = _unflatten_dense_tensors(flat_apex, unflat_t)
for i in range(len(unflat_t)):
    assert torch.eq(unflat_t[i], unflat_apex[i]).all()


# the programs being tested
def py():
    for i in range(1000):
        unflat = _unflatten_dense_tensors(flat_t, unflat_t)


def cpp():
    for i in range(1000):
        unflat = unflatten(flat_t, unflat_t)


def apex():
    for i in range(1000):
        unflat = unflatten_apex(flat_t, unflat_t)


#### cProfile ####

import cProfile


def cprofileme():
    print("--------------- cProfile -----------------")
    print("py")
    cProfile.run("py()", sort=-1)
    gc.collect()
    get_accelerator().empty_cache()
    print("cpp")
    cProfile.run("cpp()", sort=-1)
    gc.collect()
    get_accelerator().empty_cache()
    print("apex")
    cProfile.run("apex()", sort=-1)
    gc.collect()
    get_accelerator().empty_cache()


#### timeit ####

import timeit


def timeme():
    print("--------------- timeit -----------------")
    print(f'py  ={timeit.Timer("py()", globals=globals()).timeit(number=1)}')
    gc.collect()
    get_accelerator().empty_cache()
    print(f'cpp ={timeit.Timer("cpp()", globals=globals()).timeit(number=1)}')
    gc.collect()
    get_accelerator().empty_cache()
    print(f'apex={timeit.Timer("apex()", globals=globals()).timeit(number=1)}')
    gc.collect()
    get_accelerator().empty_cache()


#### line_profiler ####
# this one requires a special way to be called
# pip install line_profiler
# kernprof -l unflatten_bench.py -l; python -m line_profiler unflatten_bench.py.lprof


def line_profileme():
    print("--------------- line_profier -----------------")
    print("py")
    profile(py)()  # noqa: F821 # type: ignore
    gc.collect()
    get_accelerator().empty_cache()
    print("cpp")
    profile(cpp)()  # noqa: F821 # type: ignore
    gc.collect()
    get_accelerator().empty_cache()
    print("apex")
    profile(apex)()  # noqa: F821 # type: ignore
    gc.collect()
    get_accelerator().empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", action='store_true')
    parser.add_argument("-c", action='store_true')
    parser.add_argument("-t", action='store_true')
    args = parser.parse_args()
    if args.l:
        line_profileme()
    elif args.c:
        cprofileme()
    elif args.t:
        timeme()
