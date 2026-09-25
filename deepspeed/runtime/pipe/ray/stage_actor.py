# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False

if HAS_RAY:

    @ray.remote
    class StageActor:
        """Ray remote actor wrapping a single pipeline stage's model layers.

        Each actor holds the local subset of a ``PipelineModule``, along
        with pipeline buffers and optimizer state. The actor exposes methods
        matching the :class:`PipelineExecutor` instruction interface. The
        driver-side :class:`RayActorExecutor` dispatches schedule instructions
        to these actors via ``ray.get(actor.method.remote(...))``.

        Communication (send/recv of activations and gradients) is handled
        externally via the transport layer. The actor exposes ``get_*`` and
        ``set_*`` methods for the executor to retrieve and set tensor state
        at each buffer position.

        Args:
            stage_id (int): This actor's pipeline stage index.
            num_stages (int): Total pipeline stages.
            model (nn.Module): The local stage model layers.
            optimizer (torch.optim.Optimizer, optional): Optimizer for this stage.
        """

        def __init__(self, stage_id, num_stages, model, optimizer=None):
            self._stage_id = stage_id
            self._num_stages = num_stages
            self._model = model
            self._optimizer = optimizer
            self._is_first = (stage_id == 0)
            self._is_last = (stage_id == num_stages - 1)

            # Pipeline buffers: inputs, outputs, labels, gradients
            self._buffers = {
                'inputs': [],
                'outputs': [],
                'labels': [],
                'gradients': [],
            }
            self._num_buffers = 0

            # Pending object refs for inter-stage communication via RayTransport
            self._pending_refs = {}

        # ------------------------------------------------------------------
        # Buffer lifecycle
        # ------------------------------------------------------------------

        def reserve_buffers(self, num_buffers):
            """Allocate slots for pipeline buffers.

            Args:
                num_buffers (int): Number of buffer slots to allocate.
            """
            if self._num_buffers >= num_buffers:
                return
            num_added = num_buffers - self._num_buffers
            for key in self._buffers:
                self._buffers[key].extend([None] * num_added)
            self._num_buffers = num_buffers

        def reset_buffers(self):
            """Clear all pipeline buffer contents for a new batch."""
            for key in self._buffers:
                self._buffers[key] = [None] * self._num_buffers

        # ------------------------------------------------------------------
        # Model operations
        # ------------------------------------------------------------------

        def forward_pass(self, buffer_id):
            """Run forward on the buffered input.

            Args:
                buffer_id (int): Index of the pipeline buffer.

            Returns:
                torch.Tensor or tuple: The output tensor(s).
            """
            inputs = self._buffers['inputs'][buffer_id]
            if inputs is None:
                raise RuntimeError(f"No input data in buffer {buffer_id}")

            if isinstance(inputs, tuple):
                output = self._model(*inputs)
            else:
                output = self._model(inputs)

            self._buffers['outputs'][buffer_id] = output
            return output

        def backward_pass(self, buffer_id):
            """Run backward pass on the buffered output.

            Args:
                buffer_id (int): Index of the pipeline buffer.
            """
            outputs = self._buffers['outputs'][buffer_id]
            if outputs is None:
                raise RuntimeError(f"No output data in buffer {buffer_id}")

            grad_outputs = self._buffers['gradients'][buffer_id]

            if isinstance(outputs, (list, tuple)):
                torch.autograd.backward(outputs, grad_tensors=grad_outputs)
            elif isinstance(outputs, torch.Tensor):
                if grad_outputs is not None:
                    outputs.backward(grad_outputs)
                else:
                    outputs.backward()
            else:
                raise TypeError(f"Unexpected output type: {type(outputs)}")

        # ------------------------------------------------------------------
        # Data loading
        # ------------------------------------------------------------------

        def load_micro_batch(self, buffer_id, inputs=None, labels=None):
            """Load a micro-batch into the pipeline buffer.

            The first stage receives ``inputs``; the last stage receives
            ``labels``. Intermediate stages are no-ops.

            Args:
                buffer_id (int): Index of the pipeline buffer.
                inputs: Input data (first stage only).
                labels: Label data (last stage only).
            """
            if self._is_first and inputs is not None:
                self._buffers['inputs'][buffer_id] = inputs
            if self._is_last and labels is not None:
                self._buffers['labels'][buffer_id] = labels

        # ------------------------------------------------------------------
        # Tensor get/set for inter-stage communication
        # ------------------------------------------------------------------

        def get_activations(self, buffer_id):
            """Return activations for transfer to the next stage.

            Args:
                buffer_id (int): Index of the pipeline buffer.

            Returns:
                The tensor(s) from ``buffers['outputs'][buffer_id]``.
            """
            outputs = self._buffers['outputs'][buffer_id]
            if outputs is None:
                raise RuntimeError(f"No output data in buffer {buffer_id}")
            return outputs

        def set_inputs(self, buffer_id, tensors):
            """Store received activations from the previous stage.

            Args:
                buffer_id (int): Index of the pipeline buffer.
                tensors: The input tensor(s) to store.
            """
            self._buffers['inputs'][buffer_id] = tensors

        def get_input_grads(self, buffer_id):
            """Return gradients w.r.t. inputs for transfer to the previous stage.

            Args:
                buffer_id (int): Index of the pipeline buffer.

            Returns:
                Gradient tensor(s), or ``None``.
            """
            inputs = self._buffers['inputs'][buffer_id]
            if inputs is None:
                return None
            if isinstance(inputs, torch.Tensor):
                return inputs.grad
            return tuple(t.grad for t in inputs) if isinstance(inputs, (list, tuple)) else None

        def set_output_grads(self, buffer_id, grads):
            """Store received output gradients from the next stage.

            Args:
                buffer_id (int): Index of the pipeline buffer.
                grads: The gradient tensor(s) to store.
            """
            self._buffers['gradients'][buffer_id] = grads

        # ------------------------------------------------------------------
        # Optimizer
        # ------------------------------------------------------------------

        def optimizer_step(self, lr_kwargs=None):
            """Perform one optimizer step and zero gradients.

            Args:
                lr_kwargs (dict, optional): Learning rate overrides.
            """
            if self._optimizer is not None:
                self._optimizer.step()
                self._optimizer.zero_grad()

        def reduce_grads(self):
            """Reduce gradients across data-parallel replicas.

            For Ray actors, gradient reduction must be handled at the
            transport/executor level since actors are in separate processes.
            This is a placeholder for future distributed allreduce support.
            """
            pass

        def reduce_tied_grads(self):
            """Reduce tied-weight gradients across pipeline stages.

            Placeholder for future cross-stage gradient sync support.
            """
            pass

        # ------------------------------------------------------------------
        # State checkpointing
        # ------------------------------------------------------------------

        def get_model_state(self):
            """Return the model state dict for checkpointing.

            Returns:
                dict: The model's ``state_dict()``.
            """
            return self._model.state_dict()

        def load_model_state(self, state_dict):
            """Load a model state dict into this actor.

            Args:
                state_dict (dict): State dict from ``get_model_state()``.
            """
            self._model.load_state_dict(state_dict)

        def get_optimizer_state(self):
            """Return the optimizer state dict for checkpointing.

            Returns:
                dict or None: The optimizer's ``state_dict()``, or ``None``
                if no optimizer is set.
            """
            if self._optimizer is not None:
                return self._optimizer.state_dict()
            return None

        def load_optimizer_state(self, state_dict):
            """Load an optimizer state dict into this actor.

            Args:
                state_dict (dict): State dict from ``get_optimizer_state()``.
            """
            if self._optimizer is not None and state_dict is not None:
                self._optimizer.load_state_dict(state_dict)

        # ------------------------------------------------------------------
        # Properties
        # ------------------------------------------------------------------

        def get_stage_id(self):
            """int: This actor's stage index."""
            return self._stage_id

        def get_num_stages(self):
            """int: Total number of pipeline stages."""
            return self._num_stages

        def is_first_stage(self):
            """bool: ``True`` if this is the first stage."""
            return self._is_first

        def is_last_stage(self):
            """bool: ``True`` if this is the last stage."""
            return self._is_last

        def get_model(self):
            """Return the wrapped model for inspection."""
            return self._model

        def _get_node_id(self):
            """Return the Ray node ID for colocation detection."""
            return ray.get_runtime_context().get_node_id()

        def _store_pending_ref(self, src_stage, ref):
            """Store a Ray object reference sent from another stage.

            Called remotely by the transport layer to deposit a pending
            tensor reference on the receiving actor.
            """
            self._pending_refs[src_stage] = ref

        def _get_pending_ref(self, src_stage):
            """Retrieve and consume a pending Ray object reference.

            Args:
                src_stage (int): Source stage ID.

            Returns:
                The Ray object reference, or None if not pending.
            """
            return self._pending_refs.pop(src_stage, None)

else:
    # Placeholder when Ray is not installed
    class StageActor:
        pass
