import torch
from deepspeed.ops.adam import DeepSpeedCPUAdam
import time

device = 'cpu'
model_size = 1 * 1024**3
group_size = [model_size, 274432]

param = [torch.nn.Parameter(torch.ones(size, device=device)) for size in group_size]
optimizer = DeepSpeedCPUAdam(param)
#torch.set_num_threads(128)
for i, p in enumerate(param):
    p.grad = torch.ones(group_size[i], device=device)
#param.grad = torch.ones(model_size, device=device)
avg = 0
for i in range(100):
    start = time.time()
    optimizer.step()
    stop = time.time()
    avg += (stop - start)
    for i, p in enumerate(param):
        p.grad = torch.ones(group_size[i], device=device) * 2
    #param.grad = torch.ones(model_size, device=device) * 2
print("Elapsed Time is ", avg / 100)
