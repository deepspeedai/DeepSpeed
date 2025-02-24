
import torch

from typing import Callable, Dict, TYPE_CHECKING, Any, Optional, Tuple

from deepspeed.ops.op_builder import TopsBuilder
import math
import torch.nn.functional as F
from torch import Tensor

inf_module = None

exp_selection_uniform_map: Dict[torch.device, Callable] = {}

@torch.jit.script
def _top_idx(source, k):
    return torch.topk(source, k=k, dim=0)[1]

@torch.jit.script
def _capacity(gates: Tensor, capacity_factor: Tensor) -> Tensor:
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    capacity = torch.ceil((num_tokens / num_experts) * capacity_factor).to(torch.int64)
    return capacity

@torch.jit.script
def _one_hot_to_float(x, num_classes):
    return F.one_hot(x, num_classes=num_classes).float()

class MoEGatingFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, 
        activations, 
        logits, 
        logits_out, 
        capacity, 
        use_rst,
        is_grad_enabled,
        scores             ,
        expert_assignment  ,
        mapped_slots       ,
        expert_offset      ,
        expert_backup_offset,
        expert_counts      ,
        mapped_expert_counts,
        expert_cumsum      ,
        top_k              ,
    ):
        kernel = inf_module.moe_gating_fwd

        n_tokens, hidden_size = activations.shape
        _, n_experts = logits.shape

        moe_input_size = n_experts * capacity * top_k
        # always cap the size to 256-divisible buffer-size!
        if moe_input_size % 256 != 0:
            moe_input_size = (256 - moe_input_size % 256) + moe_input_size
        
        moe_input = torch.zeros(
            moe_input_size,
            hidden_size, 
            dtype=activations.dtype, 
            device=activations.device
        )
        if not is_grad_enabled:
            expert_counts.zero_()
            mapped_expert_counts.zero_()
            expert_cumsum.zero_()

            if top_k > 1:    
                torch_capacity = _capacity(logits, torch.tensor(top_k))
                # Create a mask for 1st's expert per token
                indices1_s = torch.argmax(logits, dim=1)
                num_experts = int(logits.shape[1])
                mask1 = F.one_hot(indices1_s, num_classes=num_experts)

                logits_w_noise = logits # + gumbel_rsample(logits.shape, device=logits.device)
                # Replace top-expert with min value
                logits_except1 = logits_w_noise.masked_fill(mask1.bool(), float("-inf"))
                indices2_s = torch.argmax(logits_except1, dim=1)
                mask2 = F.one_hot(indices2_s, num_classes=num_experts)

                if top_k > 2:
                    # do the same for 3 and 4
                    logits_w_noise_2 = logits_except1
                    logits_except1_2 = logits_w_noise_2.masked_fill(mask2.bool(), float("-inf"))
                    indices3_s = torch.argmax(logits_except1_2, dim=1)
                    mask3 = torch.nn.functional.one_hot(indices3_s, num_classes=n_experts)

                    logits_w_noise_3 = logits_except1_2
                    logits_except1_2_3 = logits_w_noise_3.masked_fill(mask3.bool(), float("-inf"))
                    indices4_s = torch.argmax(logits_except1_2_3, dim=1)
                    mask4 = torch.nn.functional.one_hot(indices4_s, num_classes=n_experts)

                # Random Token Selection
                uniform = exp_selection_uniform_map.get(logits.device)
                if uniform is None:
                    uniform = torch.distributions.uniform.Uniform(low=torch.tensor(0.0, device=logits.device),
                                                                high=torch.tensor(1.0, device=logits.device)).rsample
                    exp_selection_uniform_map[logits.device] = uniform
                mask1_rand = mask1 * uniform(mask1.shape)
                mask2_rand = mask2 * uniform(mask2.shape)
                if top_k > 2:
                    mask3_rand = mask3 * uniform(mask3.shape)
                    mask4_rand = mask4 * uniform(mask4.shape)
                  
                top_idx1 = _top_idx(mask1_rand, torch_capacity)
                top_idx2 = _top_idx(mask2_rand, torch_capacity)
                if top_k > 2:
                    top_idx3 = _top_idx(mask3_rand, torch_capacity)
                    top_idx4 = _top_idx(mask4_rand, torch_capacity)

                mask1 = mask1 * torch.zeros_like(mask1).scatter_(0, top_idx1, 1)
                mask2 = mask2 * torch.zeros_like(mask2).scatter_(0, top_idx2, 1)
                
                if top_k > 2:
                    mask3 = mask3 * torch.zeros_like(mask3).scatter_(0, top_idx3, 1)
                    mask4 = mask4 * torch.zeros_like(mask4).scatter_(0, top_idx4, 1)

                # Compute locations in capacity buffer
                locations1 = torch.cumsum(mask1, dim=0) - 1
                locations2 = torch.cumsum(mask2, dim=0) - 1
                
                # Update 2nd's location by accounting for locations of 1st
                locations2 += torch.sum(mask1, dim=0, keepdim=True)
                if top_k > 2:
                    locations3 = torch.cumsum(mask3, dim=0) - 1
                    locations4 = torch.cumsum(mask4, dim=0) - 1
                    locations3 += torch.sum(mask1, dim=0, keepdim=True) + torch.sum(mask2, dim=0, keepdim=True)
                    locations4 += torch.sum(mask1, dim=0, keepdim=True) + torch.sum(mask2, dim=0, keepdim=True) + torch.sum(mask3, dim=0, keepdim=True)

                # Remove locations outside capacity from mask
                mask1 *= torch.lt(locations1, torch_capacity)
                mask2 *= torch.lt(locations2, torch_capacity)
                # Store the capacity location for each token
                locations1_s = torch.sum(locations1 * mask1, dim=1)
                locations2_s = torch.sum(locations2 * mask2, dim=1)
                if top_k > 2:
                    mask3 *= torch.lt(locations3, torch_capacity)
                    mask4 *= torch.lt(locations4, torch_capacity)
                    locations3_s = torch.sum(locations3 * mask3, dim=1)
                    locations4_s = torch.sum(locations4 * mask4, dim=1)
                      
                if top_k > 2:
                    expert_offset = torch.cat([locations1_s.to(torch.int32), 
                                                locations2_s.to(torch.int32),
                                                locations3_s.to(torch.int32),
                                                locations4_s.to(torch.int32)]).contiguous()
                    expert_backup_offset = torch.cat([mask1.sum(dim=1).to(torch.int32), 
                                                        mask2.sum(dim=1).to(torch.int32), 
                                                        mask3.sum(dim=1).to(torch.int32), 
                                                        mask4.sum(dim=1).to(torch.int32)]).contiguous()
                else:
                    expert_offset = torch.cat([locations1_s.to(torch.int32), 
                                                locations2_s.to(torch.int32)]).contiguous()
                    expert_backup_offset = torch.cat([mask1.sum(dim=1).to(torch.int32), 
                                                        mask2.sum(dim=1).to(torch.int32)]).contiguous()
            kernel(
                moe_input, 
                expert_cumsum, 
                mapped_slots, 
                activations, 
                expert_counts, 
                mapped_expert_counts,
                scores, 
                expert_assignment, 
                expert_offset, 
                expert_backup_offset, 
                logits, 
                logits_out,
                top_k,
                capacity, 
                use_rst,
            )
        else:
            inf_module.moe_gating_scatter(
                moe_input, 
                expert_cumsum, 
                mapped_slots, 
                activations, 
                expert_counts, 
                mapped_expert_counts,
                scores, 
                expert_assignment, 
                expert_offset, 
                expert_backup_offset, 
                top_k,
                capacity, 
                use_rst,
            )
    
        ctx.top_k = top_k
        ctx.capacity = capacity
        ctx.use_rst = use_rst

        if is_grad_enabled:
            ctx.save_for_backward(expert_assignment, expert_offset, logits_out, mapped_slots)

        return moe_input, scores, logits_out, expert_counts, mapped_slots, expert_assignment, expert_offset, expert_backup_offset

    @staticmethod
    def backward(ctx, moe_inp_grad, scores_grad, logits_grad, expert_counts_grad, mapped_slots_grad, expert_assignment_grad, expert_offset_grad, expert_backup_offset_grad):
        (expert_assignment, 
         expert_offset, 
         logits,
         mapped_slots) = ctx.saved_tensors


        moe_inp_grad = moe_inp_grad.contiguous()
        scores_grad = scores_grad.contiguous()
        logits_grad = logits_grad.contiguous()
        
        _, hidden_size = moe_inp_grad.shape
        top_k_tokens = scores_grad.shape[0]
        n_tokens = top_k_tokens // ctx.top_k
        kernel = inf_module.moe_gating_bwd


        activations_grad = torch.zeros(n_tokens, hidden_size, dtype=moe_inp_grad.dtype, device=torch.cuda.current_device())
        kernel(
           moe_inp_grad,
           scores_grad, 
           activations_grad,
           logits_grad,
           logits,
           expert_assignment,
           expert_offset,
           mapped_slots,
           ctx.top_k,
           ctx.capacity, 
           ctx.use_rst
        )
        return activations_grad, logits_grad, logits_grad, None, None, None, scores_grad, expert_assignment_grad, mapped_slots_grad, None, None, expert_counts_grad, None, None, None

class MoEGating(torch.nn.Module):
    """
    CUDA implementation of top-1 gating. This will perform a softmax on the logits,
    and return the scale as well as its idx within that expert's allocation.
    """


    def __init__(self, 
                 logit_dtype=torch.bfloat16, 
                 n_tokens=16384, 
                 hidden_size=3072, 
                 n_experts=64,
                 top_k=1,
                 use_floored_capacity=True,
                 compute_aux_loss=False,
                 use_act_ckpting=False) -> None:
        super(MoEGating, self).__init__()
        global inf_module
        if inf_module is None:
            inf_module = TopsBuilder().load()
        self.scores             = torch.empty(n_tokens * top_k, dtype=torch.float32, device=torch.cuda.current_device())
        self.expert_assignment  = torch.empty(n_tokens * top_k, dtype=torch.int32, device=torch.cuda.current_device())

        self.mapped_slots       = torch.empty(n_tokens * top_k, dtype=torch.int32, device=torch.cuda.current_device())
        self.expert_offset      = torch.empty(n_tokens * top_k, dtype=torch.int32, device=torch.cuda.current_device())
        self.expert_backup_offset      = torch.empty(n_tokens * top_k, dtype=torch.int32, device=torch.cuda.current_device())

        self.expert_counts      = torch.empty(n_experts * top_k, dtype=torch.int32, device=torch.cuda.current_device())
        self.mapped_expert_counts = torch.empty(n_experts * top_k, dtype=torch.int32, device=torch.cuda.current_device())
        self.expert_cumsum      = torch.empty(n_experts * top_k, dtype=torch.int32, device=torch.cuda.current_device())
        self.logits             = torch.empty(n_tokens, n_experts, dtype=torch.float32, device=torch.cuda.current_device())
        self.top_k = top_k
        self.use_floored_capacity = use_floored_capacity
        self.compute_aux_loss = compute_aux_loss
        self.use_act_ckpting = use_act_ckpting

    def forward(self, 
                activations: torch.Tensor, 
                logits: torch.Tensor, 
                capacity_factor: float, 
                use_rst: bool = True
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform top_1_gating and token scatter.
        """

        n_tokens = activations.shape[0]
        n_experts = logits.shape[-1]

        capacity = int(capacity_factor * (n_tokens / n_experts)) if self.use_floored_capacity else math.ceil(capacity_factor * (n_tokens / n_experts))

        is_grad_enabled = self.use_act_ckpting and torch.is_grad_enabled()

        (moe_input, 
        self.scores, 
        self.logits, 
        self.expert_counts, 
        self.mapped_slots, 
        self.expert_assignment, 
        self.expert_offset, 
        self.expert_backup_offset) = MoEGatingFunction.apply(
            activations, 
            logits, 
            self.logits,
            capacity, 
            use_rst,
            is_grad_enabled,
            self.scores             ,
            self.expert_assignment  ,
            self.mapped_slots       ,
            self.expert_offset      ,
            self.expert_backup_offset,
            self.expert_counts      ,
            self.mapped_expert_counts,
            self.expert_cumsum      ,
            self.top_k
        )
        if self.compute_aux_loss:
            if self.top_k == 1:
                l_aux = (torch.mean(self.logits, dim=0) * self.expert_counts[: n_experts] / n_tokens).sum() * n_experts
            else:
                l_aux = torch.mean((torch.mean(self.logits, dim=0) * self.expert_counts[: n_experts] / n_tokens)) * n_experts * n_experts

            return l_aux, moe_input, self.scores, self.mapped_slots
        else:
            return self.logits, moe_input, self.scores, self.mapped_slots