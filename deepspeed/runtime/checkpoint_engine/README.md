# Checkpoint Engine


The `CheckpointEngine` was designed to modularized the checkpoint serialization. In this way, we can simply replace/refine the checkpoint serialization methods.

### Interface for `CheckpointEngine`

Basically, for checkpoint management(save/load by deepspeed with the given tag), the `CheckpointEngine` will:

	1. To make preliminaries ready by call `create(tag)`. For `torch`, we can just log some extra info as `torch` can directly call `save/load` without other preparation.

	2. After the `create(tag)`, deepspeed can call `save/load` to persist files into disk/memory/etc.

	3. When all the files for a tag are ready, deepspeed engine will call `commit()` to tell the checkpoint engine current checkpoint is complete. For original torch, it also plays the role of logger.


```python
class CheckpointEngine(object):
    # init checkpoint engine for save/load
    def __init__(self, config_params=None):
        pass

    def create(self, tag):
        # create checkpoint on give tag for save/load.
        pass

    def save(self, state_dict, path: str):
        pass

    def load(self, path: str, map_location=None):
        pass

    def commit(self, tag):
        # to tell checkpoint services if all files are ready.
        pass

```
