# Ray-Backed Pipeline Parallelism Tutorial

Progressive examples for running DeepSpeed pipeline parallelism on Ray, from
simple homogeneous setups to full heterogeneous Mixture-of-Experts pipelines.

## Overview

DeepSpeed's Ray pipeline infrastructure (`deepspeed/runtime/pipe/ray/`) lets
you distribute pipeline stages across Ray actors, each with independent
resource allocation. This enables:

- **Heterogeneous placement** — different stages on different hardware
  (CPU, GPU type A, GPU type B, NPU, TPU)
- **Cross-accelerator communication** — TCP sockets bridge hardware that
  can't share an NCCL communicator
- **Per-stage resource tuning** — allocate optimal CPU cores, GPU memory,
  and custom resources per stage

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| `RayActorExecutor` | `ray_executor.py` | Dispatches pipeline instructions to per-stage Ray actors |
| `RayTransport` | `ray_transport.py` | Ray object store for inter-stage tensor transfer |
| `TcpTransport` | `tcp_transport.py` | TCP sockets for cross-platform communication |
| `ShmTransport` | `shm_transport.py` | Shared memory for CPU-CPU zero-copy transfer |
| `RayTopology` | `placement.py` | Placement group mapping with per-stage resource bundles |
| `StageActor` | `stage_actor.py` | Ray remote actor holding model layers, optimizer, and buffers |

## Example Index

| # | Example | Difficulty | Transport | Stages | Key Concept |
|---|---------|------------|-----------|--------|-------------|
| 01 | [Simple Two-Stage](01_simple_two_stage.py) | Beginner | Ray object store | 2 homogeneous GPUs | Basic Ray pipeline setup |
| 02 | [GPU + CPU Hybrid](02_gpu_cpu_hybrid.py) | Intermediate | Shared memory | 1 CPU + 1 GPU | Heterogeneous RayTopology bundles |
| 03 | [Multi-Accelerator](03_multi_accelerator.py) | Intermediate | TCP sockets | 1 GPU + 1 CPU (NPU sim) | Cross-platform tensor transfer |
| 04 | [MoE Heterogeneous](04_moe_heterogeneous.py) | Advanced | Mixed (shm + ray) | CPU + GPU + GPU | All concepts combined with MoE architecture |

## Quick Start

```bash
# Install dependencies
pip install deepspeed ray torch

# Run each example (Ray will start automatically)
python examples/ray_pipeline/01_simple_two_stage.py
python examples/ray_pipeline/02_gpu_cpu_hybrid.py
python examples/ray_pipeline/03_multi_accelerator.py
python examples/ray_pipeline/04_moe_heterogeneous.py
```

## Example 01: Simple Two-Stage Homogeneous

**Difficulty:** Beginner | **Transport:** Ray object store

The simplest starting point: two identical GPU stages communicating through
Ray's distributed object store. Shows the minimal configuration needed for
Ray pipeline parallelism.

**Configuration:**
```json
{
  "pipeline": {
    "executor": "ray",
    "transport": "ray",
    "stages": 2,
    "partition": "uniform"
  }
}
```

**What it demonstrates:**
- `pipeline.executor = "ray"` enables per-stage Ray actors
- `pipeline.transport = "ray"` uses Ray object store for tensor transfer
- Uniform model partitioning across two stages

## Example 02: GPU + CPU Hybrid

**Difficulty:** Intermediate | **Transport:** Shared memory

Places the embedding stage on CPU (cheap) and the transformer stage on GPU
(compute). Uses `RayTopology` with custom resource bundles to control
hardware allocation per stage.

**Configuration:**
```json
{
  "pipeline": {
    "executor": "ray",
    "transport": "shm",
    "stages": 2,
    "partition": "parameters"
  }
}
```

**Bundles:**
```python
[
    {"CPU": 4, "GPU": 0},   # Stage 0: Embedding on CPU
    {"CPU": 1, "GPU": 1},   # Stage 1: Transformer on GPU
]
```

**What it demonstrates:**
- Custom `RayTopology` per-stage resource bundles
- `ShmTransport` for zero-copy CPU-CPU transfers
- Parameter-balanced partitioning (`"partition": "parameters"`)

## Example 03: Multi-Accelerator

**Difficulty:** Intermediate | **Transport:** TCP

Simulates a cross-accelerator setup where Stage 0 runs on a GPU and Stage 1
runs on a NPU (simulated with CPU). TCP transport bridges the two hardware
types.

**Configuration:**
```json
{
  "pipeline": {
    "executor": "ray",
    "transport": "tcp",
    "tcp_send_port": 20000,
    "tcp_recv_port": 20001,
    "tcp_host": "127.0.0.1",
    "stages": 2,
    "partition": "parameters"
  }
}
```

**What it demonstrates:**
- `TcpTransport` config with send/recv port assignment
- Cross-accelerator pipeline (GPU -> simulated NPU)
- Tensor serialization over TCP for non-NCCL hardware

## Example 04: MoE Heterogeneous

**Difficulty:** Advanced | **Architecture:** 3-stage MoE

A full Mixture-of-Experts pipeline with heterogeneous placement. Three
stages run on different hardware types, combining all patterns from the
previous examples.

**Architecture:**
```
Stage 0: Embedding  .............. CPU (SharedMemory transport)
Stage 1: Self-attention  ......... GPU (Ray object store)
Stage 2: Router + Expert FFNs .... GPU (Ray object store)
```

**Configuration:**
```json
{
  "pipeline": {
    "executor": "ray",
    "transport": "ray",
    "stages": 3,
    "partition": "parameters"
  }
}
```

**Bundles:**
```python
[
    {"CPU": 4, "GPU": 0},   # Stage 0: Embedding on CPU
    {"CPU": 1, "GPU": 1},   # Stage 1: Attention on GPU
    {"CPU": 1, "GPU": 1},   # Stage 2: MoE experts on GPU
]
```

**What it demonstrates:**
- MoE architecture with router, multiple experts, and expert dispatch
- Three-stage heterogeneous pipeline with custom bundles
- All transport types in a single pipeline
- Production-adjacent expert routing pattern

## Configuration Reference

### Transport Backends

| Transport | Backend Key | Use Case | Requires GPU? |
|-----------|------------|----------|---------------|
| Ray object store | `"ray"` | Same-cluster GPU-GPU or heterogeneous | No |
| TCP sockets | `"tcp"` | Cross-platform, different accelerators | No |
| Shared memory | `"shm"` | Same-node CPU-CPU (zero-copy) | No |

### RayTopology Bundles

Each bundle is a dict of Ray resource labels. Common patterns:

```python
# Homogeneous GPU
[{"GPU": 1, "CPU": 1}, {"GPU": 1, "CPU": 1}]

# CPU + GPU hybrid
[{"CPU": 4, "GPU": 0}, {"CPU": 1, "GPU": 1}]

# Multi-GPU types (requires Ray custom resources)
[{"GPU_A": 1}, {"GPU_B": 1}]
```

### Pipeline Partition Strategies

| Strategy | Key | Description |
|----------|-----|-------------|
| Uniform | `"uniform"` | Split layers evenly across stages |
| Parameters | `"parameters"` | Balance by parameter count (better for heterogeneous) |

## Troubleshooting

### Ray init fails with "No GPUs found"

If you don't have GPUs, examples 01 and 03 require GPUs. Either:
- Run on a machine with GPUs
- For development, skip GPU-dependent examples and start with example 02
  (which runs Stage 0 on CPU)

### "Ray is not installed"

```bash
pip install ray
```

### TCP port conflicts

If you see `Address already in use` for TCP transport:
- Change `tcp_send_port` and `tcp_recv_port` to unused ports
- Kill stale processes: `pkill -f "python examples/ray_pipeline"`

### "No actor handle registered for stage X"

This error occurs if the Ray topology wasn't properly initialized. Ensure:
- Ray is started with `ray.init()` before running the pipeline
- The `pipeline.executor` is set to `"ray"` in the config

## Next Steps

After completing these examples:

1. **Add real NPU/TPU resources** - Replace simulated CPU stages with
   actual accelerator types via Ray custom resources (`--resources='{"NPU": 1}'`)
2. **Scale experts** - Use DeepSpeed's MoE module for efficient expert
   parallelism and load-balanced routing
3. **Production deployment** - Add checkpoint save/load using
   `StageActor.get_model_state()` and `load_model_state()`
4. **Performance tuning** - Experiment with partition strategies, bundle
   sizes, and transport backends for your hardware topology
