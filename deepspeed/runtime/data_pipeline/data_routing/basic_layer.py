# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed.utils import logger
from torch import Tensor
from torch.nn import Module
from ..constants import *
from deepspeed.ops.random_ltd.dropping_utils import gpt_sample_tokens, bert_sample_tokens, GatherTokens, ScatterTokens


#####based on the paper random-ltd: https://arxiv.org/abs/2211.11586
class RandomLayerTokenDrop(Module):
    """
    A  layer wrapper for random LTD
    """

    def __init__(self, layer: Module):
        super(RandomLayerTokenDrop, self).__init__()
        self.random_ltd_layer = layer
        self.reserved_length = None  #config['max_value']
        self.random_ltd_scheduler = None
        self.max_length = None
        self.reserved_length = -1
        self.curr_seq = -1
        self.batch_first = False

    def init_config(self, config, scheduler, random_ltd_layer_id):
        self.random_ltd_scheduler = scheduler
        self.random_ltd_layer_id = random_ltd_layer_id
        self.max_length = self.random_ltd_scheduler.state[RANDOM_LTD_MAX_VALUE]

        self.mask_name = config[RANDOM_LTD_MODEL_MASK_NAME]
        self.micro_bs = config[RANDOM_LTD_MICRO_BATCH_SIZE]
        self.random_ltd_num_layer = self.random_ltd_scheduler.random_ltd_layer_num
        hs_order = config[RANDOM_LTD_HIDDEN_STATE_ORDER]
        self.model_type = config[RANDOM_LTD_MODEL_TYPE]

        if hs_order == 'batch_seq_dim':
            self.get_hidden_tensor_shape = self.get_bsh
            self.batch_first = True
        elif hs_order == 'seq_batch_dim':
            self.get_hidden_tensor_shape = self.get_sbh
            self.batch_first = False
        else:
            logger.warning(
                "************For now, we only support batch_seq_dim or seq_batch_dim inputs. You can easily \
                     your own input dimension orders************")
            raise NotImplementedError

        if self.model_type == 'encoder':
            self.index_generator = bert_sample_tokens
        elif self.model_type == 'decoder':
            self.index_generator = gpt_sample_tokens
        else:
            logger.warning("************For now, we only support encoder-only or decoder-only models************")
            raise NotImplementedError

    def get_bsh(self, hidden_stats):
        self.curr_seq, self.curr_micro_batch = hidden_stats.size()[1], hidden_stats.size()[0]

    def get_sbh(self, hidden_stats):
        self.curr_seq, self.curr_micro_batch = hidden_stats.size()[0], hidden_stats.size()[1]

    def forward(self, hidden_states, **kwargs) -> Tensor:
        if self.random_ltd_scheduler is not None:
            self.reserved_length = self.random_ltd_scheduler.get_current_seq()
            self.get_hidden_tensor_shape(hidden_states)
        if self.training and self.random_ltd_scheduler is not None and self.reserved_length < self.curr_seq:
            if self.mask_name is not None:
                mask = kwargs[self.mask_name]
            else:
                mask = None
            if self.random_ltd_layer_id == 0:
                sampled_indices, part_attention_mask = self.index_generator(self.reserved_length,\
                                                                                      self.curr_seq, \
                                                                                      self.curr_micro_batch, \
                                                                                      self.random_ltd_num_layer, \
                                                                                      hidden_states.device, mask)
                self.random_ltd_scheduler.state[RANDOM_LTD_SAMPLE_INDEX] = sampled_indices
                self.random_ltd_scheduler.state[RANDOM_LTD_ATTENTION_MASK] = part_attention_mask
            else:
                sampled_indices = self.random_ltd_scheduler.state[RANDOM_LTD_SAMPLE_INDEX]
                part_attention_mask = self.random_ltd_scheduler.state[RANDOM_LTD_ATTENTION_MASK]

            hidden_states, part_hidden_states = GatherTokens.apply(hidden_states,
                                                                   sampled_indices[self.random_ltd_layer_id, :, :],
                                                                   self.batch_first)
            if self.mask_name is not None:
                if self.model_type == 'encoder':
                    kwargs[self.mask_name] = part_attention_mask[self.random_ltd_layer_id]
                else:
                    kwargs[self.mask_name] = part_attention_mask

            outputs = self.random_ltd_layer(part_hidden_states, **kwargs)

            if isinstance(outputs, tuple):
                hidden_states = ScatterTokens.apply(hidden_states, outputs[0],
                                                    sampled_indices[self.random_ltd_layer_id, :, :], self.batch_first)
                my_list = list(outputs)
                my_list[0] = hidden_states
                return tuple(my_list)
            elif isinstance(outputs, Tensor):
                hidden_states = ScatterTokens.apply(hidden_states, outputs,
                                                    sampled_indices[self.random_ltd_layer_id, :, :], self.batch_first)
                return hidden_states
            else:
                logger.warning("************For now, we only support tuple and tensor output.  \
                       You need to adjust the output according to the layer in your model************")
                raise NotImplementedError
        else:
            return self.random_ltd_layer(hidden_states, **kwargs)
