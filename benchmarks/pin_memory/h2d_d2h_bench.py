# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Compare torch and native pinned-memory H2D/D2H bandwidth on one CUDA GPU."""

import argparse
import json
import os
import subprocess
import sys

import torch

from deepspeed.accelerator import get_accelerator

ARMS = {
    "torch": {
        "DS_PIN_MEMORY_BACKEND": "torch",
        "DS_PIN_MEMORY_REGISTER_DEVICE": "1"
    },
    "native-unregistered": {
        "DS_PIN_MEMORY_BACKEND": "native",
        "DS_PIN_MEMORY_REGISTER_DEVICE": "0"
    },
    "native-registered": {
        "DS_PIN_MEMORY_BACKEND": "native",
        "DS_PIN_MEMORY_REGISTER_DEVICE": "1"
    },
}


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--sizes-mib", type=int, nargs="+", default=[4, 64, 256])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    return parser.parse_args()


def _time_copy(accelerator, copy_fn, stream, warmup, iters):
    with accelerator.stream(stream):
        for _ in range(warmup):
            copy_fn()
        stream.synchronize()
        start = accelerator.Event(enable_timing=True)
        end = accelerator.Event(enable_timing=True)
        start.record(stream)
        for _ in range(iters):
            copy_fn()
        end.record(stream)
    stream.synchronize()
    return start.elapsed_time(end) / 1000.0 / iters


def _allocate_host(accelerator, numel, arm):
    if arm == "torch":
        return torch.empty(numel, dtype=torch.float32, pin_memory=True)

    return accelerator.pin_memory(torch.empty(numel, dtype=torch.float32), make_copy=False)


def _run_arm(args):
    for key, value in ARMS[args.arm].items():
        os.environ[key] = value

    accelerator = get_accelerator()
    if accelerator.device_name() != "cuda" or not accelerator.is_available():
        raise RuntimeError("CUDA GPU is required")
    accelerator.set_device(0)
    stream = accelerator.Stream()

    for size_mib in args.sizes_mib:
        num_bytes = size_mib * 1024 * 1024
        numel = num_bytes // torch.tensor([], dtype=torch.float32).element_size()
        host = _allocate_host(accelerator, numel, args.arm)
        device = torch.empty_like(host, device=accelerator.current_device_name())

        h2d_seconds = _time_copy(accelerator, lambda: device.copy_(host, non_blocking=True), stream, args.warmup,
                                 args.iters)
        d2h_seconds = _time_copy(accelerator, lambda: host.copy_(device, non_blocking=True), stream, args.warmup,
                                 args.iters)

        result = {
            "arm": args.arm,
            "size_mib": size_mib,
            "h2d_gbps": num_bytes / h2d_seconds / 1e9,
            "d2h_gbps": num_bytes / d2h_seconds / 1e9,
            "torch_is_pinned": host.is_pinned(),
            "accelerator_is_pinned": accelerator.is_pinned(host),
        }
        print(f"RESULT={json.dumps(result, sort_keys=True)}", flush=True)
        accelerator.unpin_memory(host)


def _run_all(args):
    results = []
    for arm in ARMS:
        command = [
            sys.executable,
            os.path.abspath(__file__),
            "--arm",
            arm,
            "--sizes-mib",
            *(str(size) for size in args.sizes_mib),
            "--warmup",
            str(args.warmup),
            "--iters",
            str(args.iters),
        ]
        process = subprocess.run(command, check=True, text=True, capture_output=True)
        if process.stderr:
            print(process.stderr, file=sys.stderr, end="")
        for line in process.stdout.splitlines():
            print(line)
            if line.startswith("RESULT="):
                results.append(json.loads(line.removeprefix("RESULT=")))

    print("\narm,size_mib,h2d_gbps,d2h_gbps,torch_is_pinned,accelerator_is_pinned")
    for result in results:
        print(f"{result['arm']},{result['size_mib']},{result['h2d_gbps']:.2f},{result['d2h_gbps']:.2f},"
              f"{result['torch_is_pinned']},{result['accelerator_is_pinned']}")


if __name__ == "__main__":
    arguments = _parse_args()
    if arguments.arm:
        _run_arm(arguments)
    else:
        _run_all(arguments)
