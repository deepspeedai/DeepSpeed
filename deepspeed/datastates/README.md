# DataStates-LLM checkpointing engine.

This feature is not enabled by default. To enable, set the following options in ds_config.json and download the [DataStates-LLM checkpointing library](https://github.com/DataStates/datastates-llm/). A detailed tutorial is available [here](../../docs/_tutorials/datastates-async-checkpointing.md).

```
{
    ... other deepspeed config options,
    "datastates_ckpt": {
        "host_cache_size": 16
	}
}
```
