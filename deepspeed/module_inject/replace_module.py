import copy
import torch
import deepspeed


def replace_transformer_layer(orig_layer_impl,
                              model,
                              micro_batch_size,
                              bert_config,
                              seed,
                              max_seq_length,
                              preln=False,
                              fp16=True,
                              huggingface=False,
                              local_rank=-1):
    """ Replace bert-style transformer layers with DeepSpeed's transformer layer
    Arguments:
        orig_layer_impl (torch.nn.Module): the original transformer layer implementation to look for,
            e.g., transformers.modeling_bert.BertLayer.
        model (torch.nn.Module): user's nn.module representing their model
        micro_batch_size (int): micro batch size per gpu used during training/eval
        bert_config (dict): model config containing hidden size, attention heads, etc.
        seed (int): random seed value
        max_seq_length (int): max sequence length for training
        preln (bool): does the original layer implementation do pre or post layer norm?
        fp16 (bool): fp16 or fp32
        huggingface (bool): huggingface implementation is unique (supports both encoder/decoder modes)

    Returns:
        Updated nn.module with replaced transformer layers
    """
    def replace_fn(child):
        transformer_config = deepspeed.DeepSpeedTransformerConfig(
            batch_size=micro_batch_size,
            max_seq_length=max_seq_length,
            hidden_size=bert_config.hidden_size,
            heads=bert_config.num_attention_heads,
            attn_dropout_ratio=bert_config.attention_probs_dropout_prob,
            hidden_dropout_ratio=bert_config.hidden_dropout_prob,
            num_hidden_layers=bert_config.num_hidden_layers,
            initializer_range=bert_config.initializer_range,
            seed=seed,
            fp16=fp16,
            pre_layer_norm=preln,
            huggingface=huggingface,
            local_rank=local_rank)
        new_module = deepspeed.DeepSpeedTransformerLayer(transformer_config)

        # copy relevant state from child -> new module
        qw = child.attention.self.query.weight
        qb = child.attention.self.query.bias
        kw = child.attention.self.key.weight
        kb = child.attention.self.key.bias
        vw = child.attention.self.value.weight
        vb = child.attention.self.value.bias

        qkvw = torch.cat((qw, kw, vw), 0)
        qkvb = torch.cat((qb, kb, vb), 0)

        #qw.data,kw.data,vw.data = torch.chunk(qkvw, 3, axis=0)
        #qb.data,kb.data,vb.data = torch.chunk(qkvb, 3, axis=0)

        new_module.attn_qkvw.data = qkvw
        new_module.attn_qkvb.data = qkvb
        new_module.attn_ow.data = child.attention.output.dense.weight
        new_module.attn_ob.data = child.attention.output.dense.bias
        if preln:
            attention_layernorm = child.PostAttentionLayerNorm
        else:
            attention_layernorm = child.attention.output.LayerNorm
        new_module.attn_nw.data = attention_layernorm.weight
        new_module.attn_nb.data = attention_layernorm.bias
        if preln:
            intermediate_ff = child.intermediate.dense_act
        else:
            intermediate_ff = child.intermediate.dense
        new_module.inter_w.data = intermediate_ff.weight
        new_module.inter_b.data = intermediate_ff.bias
        new_module.output_w.data = child.output.dense.weight
        new_module.output_b.data = child.output.dense.bias
        if preln:
            transformer_layernorm = child.PreAttentionLayerNorm
        else:
            transformer_layernorm = child.output.LayerNorm
        new_module.norm_w.data = transformer_layernorm.weight
        new_module.norm_b.data = transformer_layernorm.bias
        return new_module

    return replace_module(model=model, orig_class=orig_layer_impl, replace_fn=replace_fn)


def revert_transformer_layer(orig_layer_impl, model, bert_config, preln=False):
    """ Revert DeepSpeed's transformer layer back to original bert-style transformer layer
    Arguments:
        orig_layer_impl (torch.nn.Module): the original transformer layer implementation that was replaced,
            e.g., transformers.modeling_bert.BertLayer.
        model (torch.nn.Module): user's nn.module representing their model
        bert_config (dict): model config containing hidden size, attention heads, etc.

    Returns:
        Updated nn.module with original bert-style transformer layers
    """
    def replace_fn(child):
        #from turing.nvidia_modelingpreln import BertLayer
        orig_module = orig_layer_impl(bert_config)

        # copy relevant state from child -> original module
        qkvw = child.attn_qkvw.data
        qkvb = child.attn_qkvb.data

        qw, kw, vw = torch.chunk(qkvw, 3, axis=0)
        qb, kb, vb = torch.chunk(qkvb, 3, axis=0)

        orig_module.attention.self.query.weight.data = qw
        orig_module.attention.self.query.bias.data = qb
        orig_module.attention.self.key.weight.data = kw
        orig_module.attention.self.key.bias.data = kb
        orig_module.attention.self.value.weight.data = vw
        orig_module.attention.self.value.bias.data = vb

        orig_module.attention.output.dense.weight.data = child.attn_ow.data
        orig_module.attention.output.dense.bias.data = child.attn_ob.data

        attn_ln_w = child.attn_nw.data
        attn_ln_b = child.attn_nb.data
        if preln:
            orig_module.PostAttentionLayerNorm.weight.data = attn_ln_w
            orig_module.PostAttentionLayerNorm.bias.data = attn_ln_b
        else:
            orig_module.attention.output.LayerNorm.weight.data = attn_ln_w
            orig_module.attention.output.LayerNorm.bias.data = attn_ln_b

        inter_ff_w = child.inter_w.data
        inter_ff_b = child.inter_b.data
        if preln:
            orig_module.intermediate.dense_act.weight.data = inter_ff_w
            orig_module.intermediate.dense_act.bias.data = inter_ff_b
        else:
            orig_module.intermediate.dense.weight.data = inter_ff_w
            orig_module.intermediate.dense.bias.data = inter_ff_b

        orig_module.output.dense.weight.data = child.output_w.data
        orig_module.output.dense.bias.data = child.output_b.data

        transformer_ln_w = child.norm_w.data
        transformer_ln_b = child.norm_b.data
        if preln:
            orig_module.PreAttentionLayerNorm.weight.data = transformer_ln_w
            orig_module.PreAttentionLayerNorm.bias.data = transformer_ln_b
        else:
            orig_module.output.LayerNorm.weight.data = transformer_ln_w
            orig_module.output.LayerNorm.bias.data = transformer_ln_b
        return orig_module

    return replace_module(model=model,
                          orig_class=deepspeed.DeepSpeedTransformerLayer,
                          replace_fn=replace_fn)


def replace_module(model, orig_class, replace_fn):
    """ Scan the model for instances of ``orig_clas:`` to replace using ``replace_fn``.
    Arguments:
        model (torch.nn.Module): the model to augment
        orig_class (torch.nn.Module): the module to search for
        replace_fn (method): a method to convert instances of ``orig_class`` to the
                             desired type and return a new instance.

    Returns:
        A modified ``model``.
    """
    policy = {orig_class: replace_fn}
    return _replace_module(model, policy)


def _replace_module(model, policies):
    """ Traverse model's children recursively and apply any transformations in ``policies``.
    Arguments:
        model (torch.nn.Module): model to augment
        policies (dict): Mapping of source class to replacement function.

    Returns:
        Modified ``model``.
    """
    for name, child in model.named_children():
        if child.__class__ in policies:
            orig = repr(child)
            setattr(model, name, policies[child.__class__](child))
            new = getattr(model, name)
        else:
            _replace_module(child, policies)

    return model
