import torch

import deepspeed
from deepspeed.tops import SwiGlu
import torch.nn.functional as F
import time 

def calc_error(ds_out, pt_out):
    error = (ds_out - pt_out).abs().float().sum() / pt_out.numel()
    rel_error = ((pt_out - ds_out).abs() / (pt_out + 1e-5).abs()).float().sum() / pt_out.numel()
    return error, rel_error

def pt_swiglu(x):
    x = torch.chunk(x, 2, dim=-1)
    return F.silu(x[0]) * x[1]
    
a = torch.ones(4 * 4096, 16384, dtype=torch.bfloat16, device=torch.cuda.current_device(), requires_grad=True)
aa = torch.ones(4 * 4096, 16384, dtype=torch.bfloat16, device=torch.cuda.current_device(), requires_grad=True)


aa.retain_grad()

swiglu = SwiGlu()
for _ in range(10):
    ds_out = swiglu(a)
    error = ds_out.sum()
    error.backward()
torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    ds_out = swiglu(a)
    error = ds_out.sum()
    error.backward()
torch.cuda.synchronize()
t1 = time.time()
ds_time = t1 - t0
print(ds_time, (a.numel() + ds_out.numel()) * 2 / ds_time / 1000000)

pt_out = pt_swiglu(aa)
for _ in range(10):
    pt_out = pt_swiglu(aa)
    error1 = pt_out.sum()
    error1.backward()
torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    pt_out = pt_swiglu(aa)
    error1 = pt_out.sum()
    error1.backward()
torch.cuda.synchronize()
t1 = time.time()
pt_time = t1 - t0
print(pt_time, (a.numel() + ds_out.numel()) * 2 / pt_time / 1000000)
print(f'speedup: {pt_time/ds_time:.2f}x')