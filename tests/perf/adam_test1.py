# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from deepspeed.ops.adam import DeepSpeedCPUAdam
import time
from deepspeed.accelerator import get_accelerator

device = 'cpu'
model_size = 1 * 1024**3
param = torch.nn.Parameter(torch.ones(model_size, device=device))
param_fp16 = torch.nn.Parameter(torch.ones(model_size, dtype=torch.half, device=get_accelerator().device_name(0)))

optimizer = DeepSpeedCPUAdam([param])
#torch.set_num_threads(128)
param.grad = torch.ones(model_size, device=device)
avg = 0
for i in range(100):
    start = time.time()
    optimizer.step(fp16_param_groups=[param_fp16])
    stop = time.time()
    avg += (stop - start)
    param.grad = torch.ones(model_size, device=device) * 2
print("Elapsed Time is ", avg / 100)
