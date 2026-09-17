import torch

import deepspeed
from deepspeed.tops import RoPE
from megatron.model.rotary_pos_embedding import RotaryEmbedding


rotary_pos_emb = RotaryEmbedding(128)
rotary_pos_emb = self.rotary_pos_emb(rotary_pos_emb_len)

rope = RoPE()
query_layer = torch.randn(4096, 1, 1, 128, device=torch.cuda.current_device(), dtype=torch.bfloat16)
key_layer = torch.randn(4096, 1, 1, 128, device=torch.cuda.current_device(), dtype=torch.bfloat16)

query_layer = apply_rotary_pos_emb(query_layer, q_pos_emb)
key_layer = apply_rotary_pos_emb(key_layer, k_pos_emb)