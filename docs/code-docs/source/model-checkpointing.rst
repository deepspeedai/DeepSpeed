Model Checkpointing
===================

DeepSpeed provides routines for checkpointing model state during training.

Loading Training Checkpoints
----------------------------
.. autofunction:: deepspeed.DeepSpeedEngine.load_checkpoint

Saving Training Checkpoints
---------------------------
.. autofunction:: deepspeed.DeepSpeedEngine.save_checkpoint


ZeRO Checkpoint fp32 Weights Recovery
-------------------------------------

DeepSpeed provides routines for extracting fp32 weights from the saved ZeRO checkpoint's optimizer states.

.. autofunction:: deepspeed.utils.zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint

.. autofunction:: deepspeed.utils.zero_to_fp32.load_state_dict_from_zero_checkpoint

.. autofunction:: deepspeed.utils.zero_to_fp32.convert_zero_checkpoint_to_fp32_state_dict


Avoiding ZeRO Checkpoint Bloat
------------------------------
ZeRO stage 1 and 2 checkpoints created using ``torch.save()`` can sometimes be larger than expected. This bloat
is caused by the interaction of ZeRO's tensor flattening and torch's tensor `storage management <https://pytorch.org/docs/stable/notes/serialization.html#preserve-storage-sharing>`_ .
You can avoid this problem by using the ``clone_tensors_for_torch_save`` utility of DeepSpeed as illustrated below.

.. autofunction:: deepspeed.checkpoint.utils.clone_tensors_for_torch_save

The following code snippet illustrates this functionality for creating a HuggingFace model checkpoint:

.. code-block:: python

    ds_config = {
     ...
    }
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-13b", torch_dtype=torch.float16)
    ds_engine, _, _, _ = deepspeed.initialize(model=model, config_params=ds_config)
    lean_state_dict = deepspeed.checkpoint.utils.clone_tensors_for_torch_save(ds_engine.module.state_dict())
    ds_engine.module.save_pretrained("lean_after", state_dict=lean_state_dict)
