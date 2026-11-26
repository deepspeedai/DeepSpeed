Pipeline Parallelism
====================

Model Specification
--------------------
.. autoclass:: deepspeed.pipe.PipelineModule
    :members:

.. autoclass:: deepspeed.pipe.DualPipeVModule
    :members:

.. autoclass:: deepspeed.pipe.LayerSpec
    :members:

.. autoclass:: deepspeed.pipe.TiedLayerSpec
    :members:

.. autoclass:: deepspeed.runtime.pipe.ProcessTopology
    :members:

Training
--------
.. automodule:: deepspeed.runtime.pipe.engine
    :members:

.. autoclass:: deepspeed.runtime.pipe.dualpipev.DualPipeVEngine
    :members:

Extending Pipeline Parallelism
------------------------------
.. automodule:: deepspeed.runtime.pipe.schedule
    :members:
