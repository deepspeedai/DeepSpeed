# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from dataclasses import dataclass
from typing import Optional

from deepspeed import comm as dist

# Attribute names that carry the key/value head count, in probe order. Modern canonical names
# first, then legacy aliases kept for older configs / checkpoints, then the query-head names as
# an implicit "kv == q heads" default for plain (non-GQA) transformers. Shared by AutoTP and the
# inference engine so the two never diverge on which model families they recognize.
_KV_HEAD_ATTRS = (
    'num_key_value_heads',  # llama-class, mistral, qwen2, gemma, phi3, dbrx top-level, ...
    'num_kv_heads',  # falcon, qwen3_moe
    'multi_query_group_num',  # chatglm2 / chatglm3 (not in stock transformers)
    'n_head_kv',  # legacy Falcon custom-code; deprecated after transformers 4.33 (-> num_kv_heads)
    'kv_n_heads',  # dbrx nested attn_config; top-level config exposes num_key_value_heads (>= transformers 4.40)
    'num_attention_heads',  # plain transformer: kv == q heads
    'n_heads',
    'attention_heads',
)


def _kv_head_count_from(config) -> Optional[int]:
    """First non-None kv-head count found on ``config`` via :data:`_KV_HEAD_ATTRS`, else ``None``."""
    for name in _KV_HEAD_ATTRS:
        value = getattr(config, name, None)
        if value is not None:
            return value
    return None


# Attribute names that carry the (query) attention head count, in probe order.
_ATTN_HEAD_ATTRS = ('num_attention_heads', 'num_heads', 'n_heads', 'n_head', 'attention_heads')


def _attention_head_count_from(config) -> Optional[int]:
    """First non-None attention head count found on ``config`` via :data:`_ATTN_HEAD_ATTRS`, else ``None``."""
    for name in _ATTN_HEAD_ATTRS:
        value = getattr(config, name, None)
        if value is not None:
            return value
    return None


@dataclass(frozen=True)
class AutoTPMeta:
    """Per-model tensor-parallel metadata AutoTP derives from the model config.

    Each model carries its own kv-head / grain values so its layers, fused-QKV repacking and
    checkpoint conversion shard consistently, and more than one AutoTP model can live in the
    same process.
    """
    num_kv_heads: Optional[int] = None
    num_attention_heads: Optional[int] = None
    n_embd: Optional[int] = None
    tp_grain_size: int = 1

    @classmethod
    def from_model_config(cls, model_config, tp_grain_size: int = 1) -> "AutoTPMeta":
        """The single source of truth for reading kv-head / hidden / attention-head counts.

        ``model_config`` may be ``None`` for callers that build a synthetic AutoTP with no real
        model (some unit tests); they get an empty meta and the resulting even-grain split.
        """
        if model_config is None:
            return cls(tp_grain_size=tp_grain_size)
        # Multimodal configs (e.g. vision-language models) keep the head counts only under
        # text_config while the outer config carries the modality heads, so descend to the
        # text config when it exists.
        text_config = getattr(model_config, "text_config", None)
        if text_config is not None:
            model_config = text_config
        num_kv_heads = _kv_head_count_from(model_config)
        n_embd = None
        for name in ('n_embd', 'hidden_size'):
            if hasattr(model_config, name):
                n_embd = getattr(model_config, name)
            if n_embd is not None:
                break
        num_attention_heads = _attention_head_count_from(model_config)
        return cls(num_kv_heads=num_kv_heads,
                   num_attention_heads=num_attention_heads,
                   n_embd=n_embd,
                   tp_grain_size=tp_grain_size)


def get_shard_size(total_size, mp_size, meta: AutoTPMeta, name=None, rank=None, mp_group=None):
    """Size of one shard of ``total_size`` split across a tensor-parallel group of ``mp_size``.

    ``meta`` carries this model's ``num_kv_heads`` / ``tp_grain_size`` so the split is stable
    for the lifetime of the model instead of depending on whichever AutoTP model was loaded
    last.

    ``rank`` is the rank *within the tensor-parallel group*, i.e. in ``[0, mp_size)``, matching
    ``dist.get_rank(group=mp_group)`` and the index used by ``get_shard_size_list``. It is not a
    global rank.
    """
    num_kv_heads = meta.num_kv_heads
    tp_grain_size = meta.tp_grain_size
    last_linear = ["lm_head", "embed_out"]
    # MoE MLP layer use near even division will get better perf.
    moe_mlp_layer = ["gate_proj", "up_proj", "down_proj", "w1", "w2", "w3"]
    not_moe_mlp_layer = True
    if name != None and any(s in str(name) for s in moe_mlp_layer):
        not_moe_mlp_layer = False
    if rank is None:
        if mp_group is not None:
            rank = dist.get_rank(group=mp_group)
        else:
            world_size = dist.get_world_size()
            if world_size != mp_size:
                raise ValueError("get_shard_size requires a group-local rank or process group when mp_size "
                                 f"({mp_size}) differs from the distributed world size ({world_size}).")
            rank = dist.get_rank()
    # A known kv-head count allows uneven division, otherwise enforce near even division.
    if num_kv_heads is not None and total_size % num_kv_heads == 0 and "mlp" not in str(name) and \
        str(name) not in last_linear and not_moe_mlp_layer:
        my_slices = (num_kv_heads // mp_size) + (1 if rank < (num_kv_heads % mp_size) else 0)
        return total_size * my_slices // num_kv_heads
    else:
        if total_size >= tp_grain_size:
            grain_size, remainder = divmod(total_size, tp_grain_size)
            shard_size = (grain_size // mp_size + (1 if rank < (grain_size % mp_size) else 0)) * tp_grain_size
            if rank == mp_size - 1:
                # Quantizing to tp_grain_size would otherwise drop total_size % tp_grain_size
                # and silently truncate the dimension. Giving that tail to the last rank keeps
                # every other rank aligned for the compute kernels.
                shard_size += remainder
            return shard_size
        else:
            return total_size // mp_size + (1 if rank < (total_size % mp_size) else 0)


def get_shard_size_list(total_size, mp_size, meta: AutoTPMeta, name=None):
    shard_sizes = []
    for i in range(mp_size):
        shard_sizes.append(get_shard_size(total_size, mp_size, meta, name, i))
    # Shards must tile the dimension exactly, otherwise the partitioned weights no longer
    # reconstruct the original tensor.
    assert sum(shard_sizes) == total_size, (
        f"AutoTP shard sizes {shard_sizes} for layer '{name}' do not sum to the dimension size "
        f"{total_size} with tp_size={mp_size}, tp_grain_size={meta.tp_grain_size} and "
        f"num_kv_heads={meta.num_kv_heads}.")
    return shard_sizes
