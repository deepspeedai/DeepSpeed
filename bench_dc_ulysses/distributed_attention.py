import os
import torch
import torch.distributed as dist
from deepspeed.sequence.layer import DistributedAttention
from sp_dp_registry import get_group, is_setup, sp_size

#TODO: Hacky, need to fix it
_padding_mask_context = None

def set_padding_mask(mask):
    global _padding_mask_context
    _padding_mask_context = mask

def get_padding_mask():
    global _padding_mask_context
    return _padding_mask_context

def clear_padding_mask():
    global _padding_mask_context
    _padding_mask_context = None

def ulysses_attention_forward(
    self,
    query_states,
    key_states,
    value_states,
    attention_mask,
    scaling=None,
    dropout=0.0,
    is_causal=True,
    **kwargs,
):
    assert is_setup(), 'Incorrectly setup SP/DP Groups.'

    gid = dist.get_rank() // sp_size()
    group = get_group(gid)

    # Ulysses expects (batch, seq, heads, dim)
    # HF standard provides (batch, heads, seq, dim)
    q = query_states.transpose(1, 2).contiguous()
    k = key_states.transpose(1, 2).contiguous()
    v = value_states.transpose(1, 2).contiguous()
    
    if not hasattr(self, "ulysses_engine"):
        self.ulysses_engine = DistributedAttention(
            sdpa_wrapper,
            group,
            scatter_idx=2, # Shard heads
            gather_idx=1   # Gather sequences
        )

    # b, s, n, h
    # Note: we don't pass attention_mask here because it's the 4D mask created by HF
    # based on sharded dimensions. We'll create the correct mask in sdpa_wrapper
    # using the original unsharded padding mask stored in context.
    attn_output = self.ulysses_engine(
        q, k, v,
        batch_dim_idx=0,
        attn_mask=None,
        dropout_p=dropout,
        is_causal=False,
        scale=scaling
    )

    # Return to HF format: (batch, seq, heads, dim) -> (batch, heads, seq, dim)
    # Note: Transformers usually expects (B, N, S, H) back, 
    # but Llama's forward handles the reshape if we are careful.
    return attn_output, None

def sdpa_wrapper(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=True, scale=None):
    # Permute from [b, s, n, h] to [b, n, s, h] for SDPA
    q = query.permute(0, 2, 1, 3).contiguous()
    k = key.permute(0, 2, 1, 3).contiguous()
    v = value.permute(0, 2, 1, 3).contiguous()
    
    # Create the attention mask from padding mask + causal mask
    padding_mask = get_padding_mask()
    combined_mask = None
    
    if padding_mask is not None:
        B, S = padding_mask.shape  # [B, S]
        device = padding_mask.device
        
        causal_mask = torch.tril(torch.ones(S, S, device=device, dtype=torch.bool))
        padding_mask_bool = (padding_mask != 0).unsqueeze(1)  # [B, 1, S]
        causal_expanded = causal_mask.unsqueeze(0)  # [1, S, S]
        combined_mask = causal_expanded & padding_mask_bool  # [B, S, S]
        combined_mask = combined_mask.unsqueeze(1)  # [B, 1, S, S]

    elif is_causal:
        pass
    
    output = torch.nn.functional.scaled_dot_product_attention(
        q, k, v,
        attn_mask=combined_mask,
        dropout_p=dropout_p,
        is_causal=(combined_mask is None and is_causal),
        scale=scale,
        enable_gqa=False
    )
    
    # Permute back from [b, n, s, h] to [b, s, n, h] for all-to-all on output
    output = output.permute(0, 2, 1, 3).contiguous()
    return output
