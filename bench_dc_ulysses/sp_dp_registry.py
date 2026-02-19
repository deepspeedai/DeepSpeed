import torch
import torch.distributed as dist

GROUP_REGISTRY = {} # int -> dist.ProcessGroup

def register_groups(groups):
    """groups: List[List[int]], e.g. [[0,1],[2,3]]"""
    for gid, ranks in enumerate(groups):
        if gid not in GROUP_REGISTRY:
            GROUP_REGISTRY[gid] = dist.new_group(ranks)

def get_group(gid: int):
    return GROUP_REGISTRY[gid] if gid is not None else dist.group.WORLD

def get_registry():
    return GROUP_REGISTRY

def is_setup():
    return GROUP_REGISTRY['is_reg'] if 'is_reg' in GROUP_REGISTRY else False

def sp_size():
    assert 'SP_SIZE' in GROUP_REGISTRY, 'SP_SIZE not init properly.'

    return GROUP_REGISTRY['SP_SIZE']

def dp_size():
    assert 'DP_SIZE' in GROUP_REGISTRY, 'DP_SIZE not init properly'

    return GROUP_REGISTRY['DP_SIZE']

def populate_registry(SP_SIZE, DP_SIZE):
    ## We register in the run_acc_lm.py file for baselines to reduce code-duplication.
    ## Else the registration happens within the SP compiler pass within deepspeed.
    group_listing = []
    offset = 0
    for _ in range(DP_SIZE):
        group_listing.append([i + offset for i in range(SP_SIZE)])
        offset += SP_SIZE 

    register_groups(group_listing)

    ## Extraneous metadata required for proper instatiation. ##
    GROUP_REGISTRY['SP_SIZE'] = SP_SIZE
    GROUP_REGISTRY['DP_SIZE'] = DP_SIZE
    GROUP_REGISTRY['is_reg'] = True
