import copy
import torch
import deepspeed

from deepspeed.ops import DeepSpeedTransformerConfig


def _copy_child_transformer_state(new_module, orig_child, pre_layer_norm):
    # copy relevant state from original child -> new module
    qw = orig_child.attention.self.query.weight
    qb = orig_child.attention.self.query.bias
    kw = orig_child.attention.self.key.weight
    kb = orig_child.attention.self.key.bias
    vw = orig_child.attention.self.value.weight
    vb = orig_child.attention.self.value.bias

    qkvw = torch.cat((qw, kw, vw), 0)
    qkvb = torch.cat((qb, kb, vb), 0)

    #qw.data,kw.data,vw.data = torch.chunk(qkvw, 3, axis=0)
    #qb.data,kb.data,vb.data = torch.chunk(qkvb, 3, axis=0)

    new_module.attn_qkvw.data = qkvw
    new_module.attn_qkvb.data = qkvb
    new_module.attn_ow.data = orig_child.attention.output.dense.weight
    new_module.attn_ob.data = orig_child.attention.output.dense.bias
    if pre_layer_norm:
        attention_layernorm = orig_child.PostAttentionLayerNorm
    else:
        attention_layernorm = orig_child.attention.output.LayerNorm
    new_module.attn_nw.data = attention_layernorm.weight
    new_module.attn_nb.data = attention_layernorm.bias
    if pre_layer_norm:
        intermediate_ff = orig_child.intermediate.dense_act
    else:
        intermediate_ff = orig_child.intermediate.dense
    new_module.inter_w.data = intermediate_ff.weight
    new_module.inter_b.data = intermediate_ff.bias
    new_module.output_w.data = orig_child.output.dense.weight
    new_module.output_b.data = orig_child.output.dense.bias
    if pre_layer_norm:
        transformer_layernorm = orig_child.PreAttentionLayerNorm
    else:
        transformer_layernorm = orig_child.output.LayerNorm
    new_module.norm_w.data = transformer_layernorm.weight
    new_module.norm_b.data = transformer_layernorm.bias


def _replace_transformer_layer(orig_layer_impl, model, transformer_config):
    """ Replace bert-style transformer layers with DeepSpeed's transformer layer
    Arguments:
        orig_layer_impl (torch.nn.Module): the original transformer layer implementation to look for,
            e.g., transformers.modeling_bert.BertLayer.
        model (torch.nn.Module): user's nn.module representing their model
        transformer_config (dict): deepspeed transformer layer config containing hidden size, attention heads, etc.
    Returns:
        Updated nn.module with replaced transformer layers
    """
    def replace_fn(child):
        new_module = deepspeed.DeepSpeedTransformerLayer(transformer_config)
        _copy_child_transformer_state(new_module,
                                      child,
                                      transformer_config.pre_layer_norm)

        return new_module

    return _replace_module(model=model,
                           orig_class=orig_layer_impl,
                           replace_fn=replace_fn)


def replace_module(orig_module_impl, model, replacement_module_config):
    """ Replace client module
    Arguments:
        orig_module_impl (torch.nn.Module): original module implementation to replace,
            e.g., transformers.modeling_bert.BertLayer.
        model (torch.nn.Module): user's nn.module representing their model
        replacement_module_config (dict): deepspeed replacement module config (e.g., DeepSpeedTransformerConfig) .

    Returns:
        Updated nn.module with replaced modules
    """
    assert isinstance(replacement_module_config, DeepSpeedTransformerConfig), \
        'Only DeepSpeedTransformerConfig is currently supported as replacement config'

    return _replace_transformer_layer(orig_layer_impl=orig_module_impl,
                                      model=model,
                                      transformer_config=replacement_module_config)


def _revert_transformer_layer(orig_layer_impl, model, bert_config, transformer_config):
    """ Revert DeepSpeed's transformer layer back to original bert-style transformer layer
    Arguments:
        orig_layer_impl (torch.nn.Module): the original transformer layer implementation that was replaced,
            e.g., transformers.modeling_bert.BertLayer.
        model (torch.nn.Module): user's nn.module representing their model
        bert_config (dict): model config containing hidden size, attention heads, etc.
        transformer_config (dict): deepspeed tranformer config used for replacement

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
        if transformer_config.pre_layer_norm:
            orig_module.PostAttentionLayerNorm.weight.data = attn_ln_w
            orig_module.PostAttentionLayerNorm.bias.data = attn_ln_b
        else:
            orig_module.attention.output.LayerNorm.weight.data = attn_ln_w
            orig_module.attention.output.LayerNorm.bias.data = attn_ln_b

        inter_ff_w = child.inter_w.data
        inter_ff_b = child.inter_b.data
        if transformer_config.pre_layer_norm:
            orig_module.intermediate.dense_act.weight.data = inter_ff_w
            orig_module.intermediate.dense_act.bias.data = inter_ff_b
        else:
            orig_module.intermediate.dense.weight.data = inter_ff_w
            orig_module.intermediate.dense.bias.data = inter_ff_b

        orig_module.output.dense.weight.data = child.output_w.data
        orig_module.output.dense.bias.data = child.output_b.data

        transformer_ln_w = child.norm_w.data
        transformer_ln_b = child.norm_b.data
        if transformer_config.pre_layer_norm:
            orig_module.PreAttentionLayerNorm.weight.data = transformer_ln_w
            orig_module.PreAttentionLayerNorm.bias.data = transformer_ln_b
        else:
            orig_module.output.LayerNorm.weight.data = transformer_ln_w
            orig_module.output.LayerNorm.bias.data = transformer_ln_b
        return orig_module

    return _replace_module(model=model,
                           orig_class=deepspeed.DeepSpeedTransformerLayer,
                           replace_fn=replace_fn)


def revert_module(orig_module_impl,
                  model,
                  orig_module_config,
                  replacement_module_config):
    """ Revert DeepSpeed's module back to original client module
    Arguments:
        orig_module_impl (torch.nn.Module): the original module that was replaced,
        e.g., transformers.modeling_bert.BertLayer.
        model (torch.nn.Module): user's nn.module representing their model
        orig_module_config (dict): original module configuration
        replacement_module_config (dict): replacement deepspeed module configuration

    Returns:
        Updated nn.module with original bert-style transformer layers
    """
    assert isinstance(replacement_module_config, DeepSpeedTransformerConfig), \
        'Only DeepSpeedTransformerConfig is currently supported as replacement config'

    return _revert_transformer_layer(orig_layer_impl=orig_module_impl,
                                     model=model,
                                     bert_config=orig_module_config,
                                     transformer_config=replacement_module_config)


def _replace_module(model, orig_class, replace_fn):
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
    return _replace_module_using_policies(model, policy)


def _replace_module_using_policies(model, policies):
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
            _replace_module_using_policies(child, policies)

    return model
