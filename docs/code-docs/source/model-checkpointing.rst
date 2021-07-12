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
