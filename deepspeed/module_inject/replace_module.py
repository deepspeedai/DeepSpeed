import copy
import torch
import deepspeed
import deepspeed.ops.transformer as transformer_inference
from .replace_policy import HFBertLayerPolicy, MegatronLayerPolicy
from .replace_policy import replace_policies
from ..constants import INFERENCE_GENERIC_MODE, INFERENCE_SPECIALIZED_MODE
from ..runtime.weight_quantizer import WeightQuantization


class ReplaceWithTensorSlicing:
    def __init__(self, mp_group=None):
        if (torch.distributed.is_initialized() and mp_group is not None):
            self.gpu_index = torch.distributed.get_rank(group=mp_group)
        else:
            self.gpu_index = 0

    def merge_assert(self, dim1, dim2):
        assert dim1 > dim2, \
            'Merging tensors is not allowed here! Please use deepspeed load_checkpoint\
            for merging your checkpoints before replacing the transformer layer with\
            inference-kerenls'

    def qkv_copy(self, dst, src):
        if src is None:
            return src
        src_shape = src.shape
        dst_shape = dst.shape

        src_split = torch.split(src, src.shape[-1] // 3, dim=-1)

        if (len(src_shape) == 2 and len(dst_shape) == 2):
            if src_shape[1] == dst_shape[1]:
                return src

            self.merge_assert(src_shape[1], dst_shape[1])
            qkv_size = dst_shape[1] // 3
            qkv_split = [torch.split(src_s, qkv_size, dim=1) for src_s in src_split]

            weight_split = [
                torch.cat([qkv_s[i] for qkv_s in qkv_split],
                          axis=1) for i in range(len(qkv_split[0]))
            ]
            dst = weight_split[self.gpu_index].to(torch.cuda.current_device())
        else:
            if src_shape[0] == dst_shape[0]:
                return src

            qkv_size = dst_shape[0] // 3
            qkv_split = [torch.split(src_s, qkv_size, dim=0) for src_s in src_split]
            bias_split = [
                torch.cat([qkv_s[i] for qkv_s in qkv_split],
                          axis=0) for i in range(len(qkv_split[0]))
            ]
            dst = bias_split[self.gpu_index].to(torch.cuda.current_device())

        return dst.contiguous()

    def copy(self, dst, src):
        if src is None:
            return src

        src_shape = src.shape
        dst_shape = dst.shape

        if (len(src_shape) == 2 and len(dst_shape) == 2):

            if src_shape[0] == dst_shape[0] and src_shape[1] == dst_shape[1]:
                return src

            if src_shape[0] != dst_shape[0]:
                self.merge_assert(src_shape[0], dst_shape[0])
                weight_split = torch.split(src, dst_shape[0])
            else:
                self.merge_assert(src_shape[1], dst_shape[1])
                weight_split = torch.split(src, dst_shape[1], dim=1)

            dst = weight_split[self.gpu_index].to(torch.cuda.current_device())
        else:
            if src_shape[0] == dst_shape[0]:
                return src

            bias_split = torch.split(src, dst_shape[-1])
            dst = bias_split[self.gpu_index].to(torch.cuda.current_device())

        return dst.contiguous()


def replace_transformer_layer(orig_layer_impl,
                              model,
                              policy=None,
                              micro_batch_size=-1,
                              config=None,
                              seed=-1,
                              hidden_size=-1,
                              num_attention_heads=-1,
                              mp_size=1,
                              mp_group=None,
                              preln=True,
                              fp16=True,
                              local_rank=-1,
                              training=True,
                              quantize=False,
                              encoder_decoder=False,
                              quantize_settings=None):
    """ Replace bert-style transformer layers with DeepSpeed's transformer layer
    Arguments:
        orig_layer_impl (torch.nn.Module): the original transformer layer implementation to look for,
            e.g., transformers.modeling_bert.BertLayer.
        model (torch.nn.Module): user's nn.module representing their model
        policy: shows the policy for mapping from the orig_layer_impl to transformer parameters
        micro_batch_size (int): micro batch size per gpu used during training/eval
        config (dict): model config containing hidden size, attention heads, etc.
        seed (int): random seed value
        max_seq_length (int): max sequence length for training
        hidden_size (int): hidden dimension
        num_attention_heads (int): numebr of attention heads
        mp_size (int): model_parallelism degree
        mp_group : model_parallel gropu initialized on the modeling side
        preln (bool): does the original layer implementation do pre or post layer norm?
        fp16 (bool): fp16 or fp32
        local_rank (int): GPU rank (optional),
        training (bool): specifying whether kernel-injection is done for training/inference (set to false for inference-mode injection)
        quantize_settings (tuple): this setting shows how we can quantize a model for running it through the inference kernels.
                It includes (quantization_scales, merge_count, mlp_extra_grouping, quantize_groups).
        encoder_decoder (bool): this flag needs to be set for huggingface Bert models.

    Returns:
        Updated nn.module with replaced transformer layers
    """
    def replace_with_policy(child, policy_cls, inference=False, preln=True, layer_id=0):
        preln = False if policy_cls is HFBertLayerPolicy else preln
        if policy_cls is HFBertLayerPolicy:
            policy = policy_cls(child, inference=inference, preln=preln)
        else:
            policy = policy_cls(child, inference=inference)

        if inference:
            hidden_size, num_attention_heads = policy.get_hidden_heads()

        attn_linear_layer, qkvw, qkvb, dense_w, dense_b, scale_attention = policy.attention()
        mlp_linear_layer, _h4h_w, _h4h_b, _4hh_w, _4hh_b = policy.mlp()
        attn_nw, attn_nb, input_nw, input_nb = policy.layerNorm()

        if quantize:
            if policy_cls is not HFBertLayerPolicy:
                qkvw = qkvw.to(torch.int8)
            dense_w = dense_w.to(torch.int8)
            _h4h_w = _h4h_w.to(torch.int8)
            _4hh_w = _4hh_w.to(torch.int8)
        elif fp16:
            qkvw = qkvw.half()
            dense_w = dense_w.half()
            _h4h_w = _h4h_w.half()
            _4hh_w = _4hh_w.half()

        if quantize or fp16:
            dense_b = dense_b.half()
            _h4h_b = _h4h_b.half()
            _4hh_b = _4hh_b.half()
            attn_nw = attn_nw.half()
            attn_nb = attn_nb.half()
            input_nw = input_nw.half()
            input_nb = input_nb.half()

        mp_replace = ReplaceWithTensorSlicing(mp_group=mp_group)

        if inference:
            transformer_config = transformer_inference.DeepSpeedInferenceConfig(
                hidden_size=hidden_size,
                heads=num_attention_heads,
                fp16=fp16,
                pre_layer_norm=preln,
                mp_size=mp_size,
                q_int8=quantize,
                encoder_decoder=(True if policy_cls is HFBertLayerPolicy else False),
                triangular_masking=(policy_cls is not HFBertLayerPolicy),
                local_attention=((config.attention_layers[layer_id] == "local")
                                 if hasattr(config,
                                            'attention_layers') else False),
                window_size=(config.window_size if hasattr(config,
                                                           'window_size') else 1))

            if quantize and quantize_settings is not None:
                (quantization_scales,
                 merge_count,
                 mlp_extra_grouping,
                 quantize_groups) = quantize_settings
                new_module = transformer_inference.DeepSpeedTransformerInference(
                    transformer_config,
                    mp_group=mp_group,
                    quantize_scales=quantization_scales[layer_id],
                    quantize_groups=quantize_groups,
                    merge_count=merge_count,
                    mlp_extra_grouping=mlp_extra_grouping,
                    qkv_merging=(policy_cls is HFBertLayerPolicy))

                if quantize and qkvw.dtype != torch.int8:
                    quantize_bits = 8
                    quantizer = WeightQuantization()
                    if policy_cls is HFBertLayerPolicy:
                        data_quantized, _ = quantizer.quantize_data(qkvw, quantize_bits, quantize_groups * 3)
                    else:
                        data_quantized, _ = quantizer.quantize_data(qkvw, quantize_bits, quantize_groups)
                    qkvw.copy_(data_quantized)
                    qkvw = qkvw.to(torch.int8)
            else:
                new_module = transformer_inference.DeepSpeedTransformerInference(
                    transformer_config,
                    mp_group=mp_group,
                )
            new_module.config.scale_attention = scale_attention

            # we want the weights in [input, output] shape
            # linear layer is created with [input, output] shape
            # transpose it here to reduce inference cost!
            def transpose(data):
                data.view(-1).copy_(data.transpose(-1, -2).contiguous().view(-1))
                data = data.reshape(data.shape[-1], data.shape[-2])
                return data

            if attn_linear_layer:
                qkvw = transpose(qkvw.data)
                dense_w = transpose(dense_w)

            if mlp_linear_layer:
                _h4h_w = transpose(_h4h_w)
                _4hh_w = transpose(_4hh_w)

            attn_block = new_module.attention
            attn_block.attn_qkvw.data = mp_replace.qkv_copy(attn_block.attn_qkvw.data,
                                                            qkvw)

            if qkvb is not None:
                qkvb = qkvb.half()
                attn_block.attn_qkvb.data = mp_replace.qkv_copy(
                    attn_block.attn_qkvb.data,
                    qkvb)
            else:
                attn_block.attn_qkvb = qkvb

            attn_block.attn_ow.data = mp_replace.copy(attn_block.attn_ow.data, dense_w)
            attn_block.attn_ob.data = mp_replace.copy(attn_block.attn_ob.data, dense_b)

            mpl_block = new_module.mlp
            mpl_block.inter_w.data = mp_replace.copy(mpl_block.inter_w.data, _h4h_w)
            mpl_block.inter_b.data = mp_replace.copy(mpl_block.inter_b.data, _h4h_b)
            mpl_block.output_w.data = mp_replace.copy(mpl_block.output_w.data, _4hh_w)
            mpl_block.output_b.data = mp_replace.copy(mpl_block.output_b.data, _4hh_b)

            new_module.mlp.attn_nw.data = attn_nw.to(torch.cuda.current_device())
            new_module.mlp.attn_nb.data = attn_nb.to(torch.cuda.current_device())
            new_module.norm_w.data = input_nw.to(torch.cuda.current_device())
            new_module.norm_b.data = input_nb.to(torch.cuda.current_device())
        else:
            transformer_config = deepspeed.DeepSpeedTransformerConfig(
                batch_size=micro_batch_size,
                hidden_size=config.hidden_size,
                heads=config.num_attention_heads,
                attn_dropout_ratio=config.attention_probs_dropout_prob,
                hidden_dropout_ratio=config.hidden_dropout_prob,
                num_hidden_layers=config.num_hidden_layers,
                initializer_range=config.initializer_range,
                seed=seed,
                fp16=fp16,
                pre_layer_norm=(False if policy_cls is HFBertLayerPolicy else preln),
                huggingface=encoder_decoder,
                local_rank=local_rank,
                stochastic_mode=True,
                normalize_invertible=True,
                training=training)
            new_module = deepspeed.DeepSpeedTransformerLayer(transformer_config)
            new_module.attn_qkvw.data = qkvw
            new_module.attn_qkvb.data = qkvb
            new_module.attn_ow.data = dense_w
            new_module.attn_ob.data = dense_b

            new_module.attn_nw.data = attn_nw
            new_module.attn_nb.data = attn_nb
            new_module.norm_w.data = input_nw
            new_module.norm_b.data = input_nb

            new_module.inter_w.data = _h4h_w
            new_module.inter_b.data = _h4h_b
            new_module.output_w.data = _4hh_w
            new_module.output_b.data = _4hh_b
        return new_module

    def replace_fn(child, _policy, layer_id=0):
        if training:
            # copy relevant state from child -> new module
            new_module = replace_with_policy(child, _policy, preln=preln)

        else:
            # copy relevant state from child -> new module
            new_module = replace_with_policy(child,
                                             _policy,
                                             inference=True,
                                             preln=(policy is not HFBertLayerPolicy),
                                             layer_id=layer_id)

        return new_module

    return replace_module(model=model,
                          orig_class=orig_layer_impl,
                          replace_fn=replace_fn,
                          _replace_policy=policy)


def revert_transformer_layer(orig_layer_impl, model, config, preln=False):
    """ Revert DeepSpeed's transformer layer back to original bert-style transformer layer
    Arguments:
        orig_layer_impl (torch.nn.Module): the original transformer layer implementation that was replaced,
            e.g., transformers.modeling_bert.BertLayer.
        model (torch.nn.Module): user's nn.module representing their model
        config (dict): model config containing hidden size, attention heads, etc.

    Returns:
        Updated nn.module with original bert-style transformer layers
    """
    def replace_fn(child, _replace_policy, layer_id):
        #from turing.nvidia_modelingpreln import BertLayer
        orig_module = orig_layer_impl(config)

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
                          replace_fn=replace_fn,
                          _replace_policy=None)


def replace_module(model, orig_class, replace_fn, _replace_policy):
    """ Scan the model for instances of ``orig_clas:`` to replace using ``replace_fn``.
    Arguments:
        model (torch.nn.Module): the model to augment
        orig_class (torch.nn.Module): the module to search for
        replace_fn (method): a method to convert instances of ``orig_class`` to the
                             desired type and return a new instance.

    Returns:
        A modified ``model``.
    """
    policy = {}
    if orig_class is not None:
        policy.update({orig_class: (replace_fn, _replace_policy)})
    else:
        for plcy in replace_policies:
            # instantiate a throw-away policy in order to populate the _orig_layer_class
            _ = plcy(None)
            if plcy._orig_layer_class is not None:
                policy.update({plcy._orig_layer_class: (replace_fn, plcy)})
    assert len(policy.items()) > 0,\
        "No default policy found! Please specifiy your policy injection_policy (like {BertLayer:HFBEertLayerPolicy})." +\
        "You can find some samples here: https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/module_inject/replace_policy.py"

    replaced_module, _ = _replace_module(model, policy)
    return replaced_module


def _replace_module(model, policies, layer_id=0):
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
            setattr(
                model,
                name,
                policies[child.__class__][0](child,
                                             policies[child.__class__][-1],
                                             layer_id))
            new = getattr(model, name)
            layer_id += 1
        else:
            _, layer_id = _replace_module(child, policies, layer_id=layer_id)

    return model, layer_id
