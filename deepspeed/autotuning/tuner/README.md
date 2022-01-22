# Tuner


`exps` is a list of experiment descriptions (dictionaries).
An experimentation description has a `ds_config` field that stores the DeepSpeed configuration to be used in the experiment.

A tuner is based on BaseTuner and at least implements the `next_batch` method. It can implement a different `tune` method from the BaseTuner's.

```python
class NewTuner(BaseTuner):
    def __init__(self, exps: list, resource_manager):
        super(NewTuner, self).__init__(exps, resource_manager)

    def next_batch(self, sample_size=1):
        pass

    def tune(self): # if it differs from BaseTuner
        pass
```
