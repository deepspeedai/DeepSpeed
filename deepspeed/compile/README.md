# DeepCompile

## Basic

You can add `compile` section to the DeepSpeed configuration file to specify the compilation parameters.
To enable DeepCompile, you need to set `deepcompile` to `true` in the `compile` section. Currently, DeepCompile is enabled only with ZeRO stage 3.
We recommend setting `sub_group_size` to a smaller value (e.g., 200000000) to reduce the peak memory by the optimizer.

```json
{
...
    "zero_optimization": {
        "stage": 3,
        "sub_group_size": 200000000
    },
    "compile": {
        "deepcompile": true,
    },
...
}
```

You can compile a model using the `compile` method of the DeepSpeed Engine.

```python
target_engine, optimizer, _, _ = deepspeed.initialize(config=config_dict,
                                                model=model,
                                                model_parameters=model.parameters())
...
target_engine.compile()
```

Here is the signature of the `compile` method. Internally, it calls `compile` API of PyTorch with the specified `backend` and `compile_kwargs`.
You can pass optimization passes to the `passes` argument as a list of strings. Each string is the name of the optimization pass.
When `passes` is `None`, DeepCompile uses the default optimization passes (We detail custom optimization passes later).

```python
def compile(self,
            backend=get_accelerator().get_compile_backend(),
            compile_kwargs={},
            schedule=None) -> None:
```

Note that the first iteration (forward and backward) may take a long time when DeepCompile is enabled.

## Custom Optimization Passes

You can register custom optimization passes to the DeepSpeed engine using the following API.
Once you register a custom optimization pass, you can use the name of the pass in `passes` argument of `compile`.

```python
target_engine.register_compile_pass(PASS_NAME, opt_pass_fn)
```

`opt_pass_fn` is a function that takes the following signature. The function should return a modified graph.

```python
def opt_pass_fn(graph: torch.fx.Graph,
                graph_id: int,
                graph_order: List[int],
                profiling_results: ProfilingResult,
                mem_budget: float,
                param_manager,
                bwd: bool,
                ds_optimizer,
                nz3) -> torch.fx.Graph:
```

Each argument has the following meaning.

- `graph`: The computation graph to optimize.
- `graph_id`: The ID of the graph.
- `graph_order`: The order of the graph.
- `profiling_results`: The profiling results of the graph.
- `mem_budget`: The memory budget for this optimization.
- `param_manager`: The parameter manager that maintains inputs/outputs of the graph.
- `bwd`: A boolean value indicating whether the graph is for the backward pass.
- `ds_optimizer`: DeepSpeed optimizer.
- `nz3`: The handle of the DeepCompile native module.

You can find examples in existing optimization passes including [adaptive prefetching](https://github.com/tohtana/DeepSpeed-internal/blob/tohtana/no_z3_hook/deepspeed/compile/passes/prefetch.py) and [selective gather](https://github.com/tohtana/DeepSpeed-internal/blob/tohtana/no_z3_hook/deepspeed/compile/passes/selective_gather.py).

You can also define a schedule for the custom optimization passes. The schedule is a list of tuples.
Each tuple consists of a step and a list of optimization passes to apply at the step.

```python
schedule = [
    (STEP_N0, [(PASS_NAME_N0_0, mem_budget_N0_0), (PASS_NAME_N0_1, mem_budget_N0_1), ...]),
    (STEP_N1, [(PASS_NAME_N1_0, mem_budget_N1_0), (PASS_NAME_N1_1, mem_budget_N1_1), ...]),
    ...
]
```

## Configuration

You can specify the following configuration items in the `compile` section.

```json
"compile": {
    "deepcompile": [false|true],
    "offload_activation": [false|true],
    "offload_opt_states": [false|true],
    "double_buffer": [false|true],
    "symmetric_memory": [false|true],
    "free_activation": [false|true],
    "dump_graphs": [false|true]
},
```

Each item has the following meaning.

- `deepcompile` (Default: `false`): Enable DeepCompile.
- `offload_activation` (Default: `false`): Offload activation tensors to the host memory.
- `offload_opt_states` (Default: `false`): Offload optimizer states to the host memory.
- `double_buffer` (Default: `false`): Enable double buffering.
- `symmetric_memory` (Default: `false`): Enable symmetric memory (Only for a single node run).
- `free_activation` (Default: `true`): Free activation tensors after the backward pass.
- `dump_graphs` (Default: `false`): Dump the computation graphs to the specified directory.
