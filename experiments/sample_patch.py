# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""TEMPORARY measurement patch (GPU box only; revert with git checkout -- .):

DS_SEGKI_SKIP=step  -> force the graph+Python-step fallback path (b=1)
DS_SAMPLE_MODE=s    -> replace Python argmax with temperature sampling
"""

P = 'deepspeed/runtime/rollout/hybrid_engine_rollout.py'
s = open(P).read()

old = '''                if (graph_op is not None and batch_size == 1 and static_logits.dtype is torch.bfloat16
                        and get_accelerator().on_accelerator(static_logits)):'''
new = '''                if (graph_op is not None and batch_size == 1 and static_logits.dtype is torch.bfloat16
                        and get_accelerator().on_accelerator(static_logits)
                        and 'step' not in __import__('os').environ.get('DS_SEGKI_SKIP', '').split(',')):'''
assert s.count(old) == 1
s = s.replace(old, new)

# both Python-step loops (b>1 path and b=1 fallback): swap argmax for sampling
old = '''            get_accelerator().replay_graph(graph)
            next_token = static_logits[:, -1, :].argmax(dim=-1, keepdim=True)
            output_ids.append(next_token)
            eos_mask |= (next_token.view(batch_size) == eos_token_id)
        return torch.cat(output_ids, dim=1)'''
new = '''            get_accelerator().replay_graph(graph)
            if 's' in __import__('os').environ.get('DS_SAMPLE_MODE', ''):
                probs = torch.softmax(static_logits[:, -1, :].float(), dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = static_logits[:, -1, :].argmax(dim=-1, keepdim=True)
            output_ids.append(next_token)
            eos_mask |= (next_token.view(batch_size) == eos_token_id)
        return torch.cat(output_ids, dim=1)'''
assert s.count(old) == 2, s.count(old)
s = s.replace(old, new)
open(P, 'w').write(s)
print("sampling patch applied (both loops)")
