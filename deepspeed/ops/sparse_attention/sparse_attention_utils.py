"""
Copyright 2020 The Microsoft DeepSpeed Team
"""

from torch import nn
from torch.nn import functional as F
from deepspeed.ops.sparse_attention import BertSparseSelfAttention, SparsityConfig
'''
This file contains few utility functions to handle adapting pretrained model with sparse self-attention module.
'''


class SparseAttentionUtils:
    """This class provides some utility functions that are use integrating sparse attention into transformer models.
    Such utilities include extending position embeddings, replacing current self-attention layer with sparse attention, padding sequences to multiple of block size, etc.

    """
    @staticmethod
    def extend_position_embedding(model, max_position):
        """This function extends the position embedding weights of a model loaded from a checkpoint.
        It assumes the new max position is bigger than the original max length.

        Arguments:
            model: required: a transformer model
            max_position: required: an integer determining new position embedding size
        Return:
            model: updated model; in which position embedding weights have been extended based on new size
        """

        if hasattr(model, 'bert'):
            original_max_position = model.bert.embeddings.position_embeddings.weight.size(
                0)
            assert max_position > original_max_position
            extend_multiples = max(1, max_position // original_max_position)
            model.bert.embeddings.position_embeddings.weight.data = model.bert.embeddings.position_embeddings.weight.repeat(
                extend_multiples,
                1)
        elif hasattr(model, 'roberta'):
            # RoBERTa has positions 0 & 1 reserved, so embedding size is max position + 2
            original_max_position, embed_size = model.roberta.embeddings.position_embeddings.weight.shape
            original_max_position -= 2
            extend_multiples = max(1, max_position // original_max_position)
            assert max_position > original_max_position
            max_position += 2
            extended_position_embedding = model.roberta.embeddings.position_embeddings.weight.new_empty(
                max_position,
                embed_size)
            k = 2
            for i in range(extend_multiples):
                extended_position_embedding[k:(
                    k + original_max_position
                )] = model.roberta.embeddings.position_embeddings.weight[2:]
                k += original_max_position
            model.roberta.embeddings.position_embeddings.weight.data = extended_position_embedding
        else:
            raise ValueError(
                'Please extend \"extend_position_embedding\" function to support your model type. It currently only supports \"bert\" & \"roberta\"!'
            )

        model.config.max_position_embeddings = max_position
        print(
            f'Extended position embeddings to {original_max_position * extend_multiples}'
        )

        return model

    @staticmethod
    def update_tokenizer_model_max_length(tokenizer, max_position):
        """This function updates the position embedding length of a tokenizer to a new max position.

        Arguments:
            tokenizer: required: a transformer tokenizer
            max_position: required: an integer determining new position embedding size
        Return:
            tokenizer: updated tokenizer; in which model maximum length has been extended based on new size
        """

        tokenizer.model_max_length = max_position
        tokenizer.init_kwargs['model_max_length'] = max_position
        print(f'updated tokenizer model max imum length to {max_position}')

        return tokenizer

    @staticmethod
    def replace_model_self_attention_with_sparse_self_attention(
        model,
        max_position,
        # SparsityConfig parameters needs to be set accordingly
        sparsity_config=SparsityConfig(num_heads=4)):
        """This function replaces the self attention layers in model encoder with sparse self attention.
        It currently supports bert and roberta model and can be easily extended to any other models following similar steps here.
        For sparsityConfig, refer to the config class.

        Arguments:
            model: required: a transformer model
            max_position: required: an integer determining new position embedding size
            sparsity_config: optional: this parameter determins sparsity pattern configuration; it is based on SparsityConfig class

        Return:
            model: updated model; in which self attention layer has been repleaced with DeepSpeed Sparse Self Attention layer.
        """

        if hasattr(model, 'bert'):
            model.config.max_position_embeddings = max_position
            replace_self_attention_layer_with_sparse_self_attention_layer(
                model.config,
                model.bert.encoder.layer,
                sparsity_config)
        elif hasattr(model, 'roberta'):
            model.config.max_position_embeddings = max_position + 2
            replace_self_attention_layer_with_sparse_self_attention_layer(
                model.config,
                model.roberta.encoder.layer,
                sparsity_config)
        else:
            raise ValueError(
                'Please extend \"update_model_self_attention_to_sparse_self_attention\" function to support \
                                     your model type. It currently only supports \"bert\" & \"roberta\"!'
            )
        return model

    @staticmethod
    def replace_self_attention_layer_with_sparse_self_attention_layer(
        config,
        layers,
        # SparsityConfig parameters needs to be set accordingly
        sparsity_config=SparsityConfig(num_heads=4)):
        """This function replaces the self attention layers in attention layer with sparse self attention.
        For sparsityConfig, refer to the config class.

        Arguments:
            config: required: transformer model config
            layers: required: transformer model attention layers
            sparsity_config: optional: this parameter determins sparsity pattern configuration; it is based on SparsityConfig class

        Return:
            layers: updated attention layers; in which self attention layers have been repleaced with DeepSpeed Sparse Self Attention layer.
        """

        for layer in layers:
            deepspeed_sparse_self_attn = BertSparseSelfAttention(config, sparsity_config)
            deepspeed_sparse_self_attn.query = layer.attention.self.query
            deepspeed_sparse_self_attn.key = layer.attention.self.key
            deepspeed_sparse_self_attn.value = layer.attention.self.value

            layer.attention.self = deepspeed_sparse_self_attn

        return layers

    @staticmethod
    def pad_to_block_size(block_size,
                          input_ids,
                          attention_mask,
                          token_type_ids,
                          position_ids,
                          inputs_embeds,
                          pad_token_id,
                          model_mbeddings):
        """This function pads input tokens and attention mask on sequence length dimension to be multiple of block size.
            This is a requirement for Sparse Transformer in which the self attention layer works on sequences of length multiple of block size.
            It needs to be called in your model, such as BertModel, right before you calculate the embedding outputs.
            Note)
            1- instead of passing your embedding layer to this function, you can simply add this function to your model. It can be more simplified if given attention_mask and/or token_type_ids are none.
            2- you need to call unpdad function before returning your model output to unpad the encoder sequence output.

            Arguments:
                block_size: required: an integer determining the block size of sparsity config.
                pad_token_id: required: an integer determining the pad token from the model config; such as bert.config.pad_token_id.
                input_ids: a torch.LongTensor of shape [batch_size, sequence_length] with the word token indices in the vocabulary
                attention_mask: a torch.LongTensor of shape [batch_size, sequence_length] with indices selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max input sequence length in the current batch. It's the mask that we typically use for attention when a batch has varying length sentences.
                token_type_ids: a torch.LongTensor of shape [batch_size, sequence_length] with the token types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to a `sentence B` token (see BERT paper for more details).
                position_ids:  a torch.LongTensor of shape [batch_size, sequence_length] with the indices of positions of each input sequence tokens in the position embeddings.
                inputs_embeds: an optional torch.FloatTensor of shape [batch_size, sequence_length, hidden_size] that contains embedded representation and can be passed instead of input_ids directly.
                model_embeddings: an optional object. If inputs_embeds are not none, this will be your model embeddings such as BertEmbeddings from your model such as BertModel. You can move this function inside your model and use self.embeddings instead of passing this parameter.

            Return:
                pad_len: an integer determining how much inputs have been padded to transfer sequence length dimension to multiple of block size.
                input_ids: if input_ids are not none padded input_ids otherwise none.
                attention_mask: if attention_mask is not none padded attention_mask otherwise none.
                token_type_ids: if token_type_ids are not none padded token_type_ids otherwise none.
                position_ids: if position_ids are not none padded position_ids otherwise none.
                inputs_embeds: if inputs_embeds are not none padded inputs_embeds otherwise none.
        """

        batch_size, seq_len = input_ids.shape if input_ids is not None else inputs_embeds.shape[:-1]

        pad_len = (block_size - seq_len % block_size) % block_size
        if pad_len > 0:
            if inputs_embeds is not None:
                pad_input_ids = inputs_embeds.new_full((batch_size,
                                                        pad_len),
                                                       pad_token_id,
                                                       dtype=torch.long)
                pad_inputs_embeds = model_embeddings(pad_input_ids)
                inputs_embeds = torch.cat([inputs_embeds, pad_inputs_embeds], dim=-2)
            # may not be needed as input_ids are not used if inputs_embeds are given
            if input_ids is not None:
                input_ids = F.pad(input_ids, (0, pad_len), value=pad_token_id)
            if position_ids is not None:
                # pad position_id with pad_token_id
                position_ids = F.pad(position_ids, (0, pad_len), value=pad_token_id)
            # pad attention mask without attention on the padding tokens
            attention_mask = F.pad(attention_mask, (0, pad_len), value=False)
            # pad token_type_ids with token_type_id = 0
            token_type_ids = F.pad(token_type_ids, (0, pad_len), value=0)

        return pad_len, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds

    @staticmethod
    def unpad_sequence_output(pad_len, sequence_output):
        """This function unpads sequence output if inputs of the model were padded.
           This is a requirement for Sparse Transformer in which the self attention layer works on sequences of length multiple of block size.
           It needs to be called in your model, such as BertModel, right before you return the model outputs.

           Arguments:
               pad_len: required: an integer determining how much model inputs have been padded to transfer sequence length dimension to multiple of block size.
               sequence_output: required: sequence output of the encoder layer.

           Return:
               sequence_output: unpaded sequence output of the encoder layer.
        """

        if (pad_len > 0):
            sequence_output = sequence_output[:, :-pad_len]
        return sequence_output
