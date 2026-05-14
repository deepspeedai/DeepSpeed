# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from copy import deepcopy
import os
import random
import numpy as np

import torch

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero import GatheredParameters

from unit.simple_model import SimpleModel
from unit.common import allclose_on_all_ranks


def compare_loss(self, config, dtype, iteration=5, hidden_dim_override=None):
    hidden_dim = hidden_dim_override if hidden_dim_override is not None else 10

    # the default tolerances of torch.testing.assert_close are too small
    RTOL = 5e-1
    ATOL = 1e-2

    # Use a fixed seed for determinism. We don't use the @enable_determinism decorator
    # because it also sets torch.use_deterministic_algorithms(True), which seems
    # incompatible with torch.compile() in test environments.
    # Might be related to https://github.com/pytorch/pytorch/issues/159855
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    seed = 123 + local_rank

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    get_accelerator().manual_seed(seed)
    get_accelerator().manual_seed_all(seed)

    device = torch.device(get_accelerator().current_device_name())
    model = SimpleModel(hidden_dim)

    i = get_accelerator().current_device()
    baseline_model = deepcopy(model)
    baseline_config = deepcopy(config)
    baseline_config["zero_optimization"]["stage"] = 0
    baseline_config["zero_optimization"]["offload_optimizer"] = {}
    baseline_engine, baseline_optimizer, _, _ = deepspeed.initialize(config=baseline_config,
                                                                     model=baseline_model,
                                                                     model_parameters=baseline_model.parameters())

    if config["zero_optimization"]["stage"] == 3:
        with deepspeed.zero.Init(config_dict_or_path=config):
            target_model = SimpleModel(hidden_dim)
        with GatheredParameters(target_model.parameters(), modifier_rank=0):
            for p1, p2 in zip(target_model.parameters(), model.parameters()):
                p1.data.copy_(p2.data)
    else:
        target_model = deepcopy(model)

    target_engine, target_optimizer, _, _ = deepspeed.initialize(config=config,
                                                                 model=target_model,
                                                                 model_parameters=target_model.parameters())
    target_engine.compile()

    train_batch_size = config["train_micro_batch_size_per_gpu"]

    xs = [torch.randn(train_batch_size, hidden_dim, device=device, dtype=dtype) for _ in range(iteration)]
    ys = [torch.randn_like(x) for x in xs]

    for x, y in zip(xs, ys):
        baseline_loss = baseline_engine(x, y)
        target_loss = target_engine(x, y)

        allclose_on_all_ranks(baseline_loss, target_loss, "Loss values are not close.", rtol=RTOL, atol=ATOL)

        baseline_engine.backward(baseline_loss)
        target_engine.backward(target_loss)

        baseline_engine.step()
        target_engine.step()

        with GatheredParameters(target_engine.parameters()):
            for p1, p2 in zip(baseline_engine.parameters(), target_engine.parameters()):
                allclose_on_all_ranks(p1, p2, "Parameters are not equal.", rtol=RTOL, atol=ATOL)

    baseline_engine.destroy()
    target_engine.destroy()


def compare_sp_loss(self, config, sp_size, iterations=3):
    """
    Compare AutoSP compiled model loss against a compiled Ulysses SP model (ground truth).

    Both engines are trained in lockstep.  After all training steps the final-step
    losses are compared.
    """
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoConfig
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from deepspeed.compile import constants as autosp_constants
    from deepspeed.compile.custom_ops.sp_dp_registry import populate_registry, get_group
    from deepspeed.sequence.layer import DistributedAttention

    RTOL, ATOL = 0.1, 0.01
    model_name = 'hf-internal-testing/tiny-random-LlamaForCausalLM'
    seq_length = 64

    torch.manual_seed(42)
    get_accelerator().manual_seed_all(42)
    device = torch.device(get_accelerator().current_device_name())

    model_config = AutoConfig.from_pretrained(model_name)
    model_config._attn_implementation = "sdpa"
    base_model = AutoModelForCausalLM.from_pretrained(model_name, config=model_config)
    vocab_size = model_config.vocab_size

    # Set up SP/DP process groups (shared by both Ulysses and AutoSP).
    dp_size = dist.get_world_size() // sp_size
    populate_registry(sp_size, dp_size)
    # The DP-rank index selects which SP group the current rank belongs to.
    sp_group = get_group(dist.get_rank() // sp_size)
    sp_rank = dist.get_rank() % sp_size
    chunk = seq_length // sp_size

    # Build a DistributedAttention wrapper that mirrors distributed_attention.py.
    # Registered under a unique key so the model's "sdpa" slot stays untouched —
    # AutoSP's graph pass can therefore find F.scaled_dot_product_attention nodes.
    def _sdpa_inner(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True, scale=None):
        # DistributedAttention delivers tensors in [b, s, n, h]; SDPA wants [b, n, s, h].
        out = F.scaled_dot_product_attention(q.permute(0, 2, 1, 3),
                                             k.permute(0, 2, 1, 3),
                                             v.permute(0, 2, 1, 3),
                                             dropout_p=dropout_p,
                                             is_causal=is_causal,
                                             scale=scale)
        return out.permute(0, 2, 1, 3)

    _dist_attn = DistributedAttention(_sdpa_inner, sp_group, scatter_idx=2, gather_idx=1)

    def _ulysses_attn_forward(module,
                              query_states,
                              key_states,
                              value_states,
                              attention_mask,
                              scaling=None,
                              dropout=0.0,
                              is_causal=False,
                              **kwargs):
        q = query_states.transpose(1, 2).contiguous()
        k = key_states.transpose(1, 2).contiguous()
        v = value_states.transpose(1, 2).contiguous()
        out = _dist_attn(q, k, v, batch_dim_idx=0, dropout_p=dropout, is_causal=is_causal, scale=scaling)
        return out, None

    ALL_ATTENTION_FUNCTIONS["ulyssess"] = _ulysses_attn_forward

    # Ulysses baseline: regular torch.compile, no deepcompile or autosp pass.
    ulysses_config = deepcopy(config)
    ulysses_config.pop("compile", None)
    ulysses_model = deepcopy(base_model)
    ulysses_model.config._attn_implementation = "ulyssess"
    ulysses_engine, _, _, _ = deepspeed.initialize(config=ulysses_config,
                                                   model=ulysses_model,
                                                   model_parameters=ulysses_model.parameters())
    ulysses_engine.compile()

    # AutoSP model: sdpa so the autosp pass can find F.scaled_dot_product_attention.
    # dynamic=True ensures all shape dimensions are treated symbolically so the autosp
    # pass can correctly shard the sequence dimension for all dtypes including fp16/bf16.
    autosp_model = deepcopy(base_model)
    autosp_engine, _, _, _ = deepspeed.initialize(config=config,
                                                  model=autosp_model,
                                                  model_parameters=autosp_model.parameters())
    autosp_engine.compile(compile_kwargs={"dynamic": True})

    # Train both engines in lockstep; compare the losses at the final step.
    ul_loss = autosp_loss = None
    for i in range(iterations):
        torch.manual_seed(42 + i)
        full_ids = torch.randint(0, vocab_size, (1, seq_length), device=device)

        # Ulysses: each rank processes its own shard.
        shard_ids = full_ids[:, sp_rank * chunk:(sp_rank + 1) * chunk]
        shard_pos = torch.arange(sp_rank * chunk, (sp_rank + 1) * chunk, device=device).unsqueeze(0)
        shard_mask = torch.ones(1, chunk, device=device, dtype=torch.long)
        ul_out = ulysses_engine(input_ids=shard_ids,
                                labels=shard_ids,
                                position_ids=shard_pos,
                                attention_mask=shard_mask)
        # Average per-shard losses across SP ranks to get the full-sequence loss.
        ul_loss = ul_out.loss.clone()
        dist.all_reduce(ul_loss, group=sp_group)
        ul_loss = ul_loss / sp_size

        # AutoSP: full sequence.  dynamic=True makes all shapes symbolic, so mark_dynamic
        # is not needed; only the tag attributes that the autosp pass uses are set here.
        autosp_ids = full_ids.clone()
        autosp_lbl = autosp_ids.clone()
        autosp_pos = torch.arange(seq_length, device=device).unsqueeze(0)
        autosp_msk = torch.ones(1, seq_length, device=device, dtype=torch.long)
        autosp_ids.tag = autosp_constants.AUTOSP_INPUT_ID_KEY
        autosp_lbl.tag = autosp_constants.AUTOSP_LABEL_ID_KEY
        autosp_pos.tag = autosp_constants.AUTOSP_POSITION_ID_KEY
        autosp_out = autosp_engine(input_ids=autosp_ids,
                                   labels=autosp_lbl,
                                   position_ids=autosp_pos,
                                   attention_mask=autosp_msk)
        autosp_loss = autosp_out.loss

        ulysses_engine.backward(ul_out.loss)
        ulysses_engine.step()
        autosp_engine.backward(autosp_loss)
        autosp_engine.step()

    allclose_on_all_ranks(autosp_loss, ul_loss, "AutoSP and Ulysses losses are not close.", rtol=RTOL, atol=ATOL)

    ulysses_engine.destroy()
    del ALL_ATTENTION_FUNCTIONS["ulyssess"]
    autosp_engine.destroy()


def create_gm_nodes(batch_size: int = 1, seq_len: int = 16):
    """
    Load a tiny LlamaForCausalLM, tag inputs with AutoSP keys, mark the sequence
    dimension dynamic, and capture the torch-fx GraphModule via a custom
    torch.compile backend.

    The returned gm is identical to what the autosp pass receives during training:
    placeholder nodes carry tensor_dict tags and meta['val'] shapes are symbolic
    (SymInt) in the sequence dimension.

    Returns:
        gm     – GraphModule with fully populated node metadata
        inputs – (input_ids, labels, position_ids) used for tracing
    """
    from transformers import AutoModelForCausalLM, AutoConfig
    from deepspeed.compile import constants

    # Each call needs a clean dynamo state; without this, the recompile_limit
    # (default 8) is exhausted across tests and the backend is never invoked.
    torch._dynamo.reset()

    model_name = 'hf-internal-testing/tiny-random-LlamaForCausalLM'
    model_config = AutoConfig.from_pretrained(model_name)
    model_config._attn_implementation = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(model_name, config=model_config)
    model.eval()

    vocab_size = model_config.vocab_size
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    position_ids = torch.arange(seq_len).unsqueeze(0)

    # dynamo propagates Python tensor attributes into node.meta['tensor_dict'];
    # find_node_by_tag relies on this to identify the AutoSP input nodes.
    input_ids.tag = constants.AUTOSP_INPUT_ID_KEY
    labels.tag = constants.AUTOSP_LABEL_ID_KEY
    position_ids.tag = constants.AUTOSP_POSITION_ID_KEY

    # Marking the sequence dim dynamic causes dynamo to emit a SymInt placeholder
    # node and store symbolic shapes in node.meta['val'], which shard_tensor_node
    # needs to locate the sequence-length symbol in the graph.
    torch._dynamo.decorators.mark_dynamic(input_ids, 1)
    torch._dynamo.decorators.mark_dynamic(labels, 1)
    torch._dynamo.decorators.mark_dynamic(position_ids, 1)

    captured_gm = [None]

    def _capture_backend(gm, example_inputs):
        if captured_gm[0] is None:
            captured_gm[0] = gm
        return gm

    compiled = torch.compile(model, backend=_capture_backend, dynamic=True)
    with torch.no_grad():
        compiled(input_ids=input_ids, labels=labels, position_ids=position_ids)

    assert captured_gm[0] is not None, "Capture backend was never invoked — graph capture failed"
    return captured_gm[0], (input_ids, labels, position_ids)


def find_sym_seq_node(gm):
    """
    Return the SymInt placeholder node for the sequence-length dimension of
    input_ids, or None if it cannot be found.
    """
    from deepspeed.compile.util import get_input_id_node
    from deepspeed.compile.fx import get_node_shape_meta, find_node_by_name

    input_ids_node = get_input_id_node(gm)
    val = get_node_shape_meta(input_ids_node)
    seq_symint = val.shape[1]
    return find_node_by_name(gm, str(seq_symint))
