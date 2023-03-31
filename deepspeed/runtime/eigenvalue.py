# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from deepspeed.utils import log_dist
import numpy as np
import logging


class Eigenvalue(object):

    def __init__(self,
                 verbose=False,
                 max_iter=100,
                 tol=1e-2,
                 stability=0,
                 gas_boundary_resolution=1,
                 layer_name='',
                 layer_num=0):
        super().__init__()

        self.verbose = verbose
        self.max_iter = max_iter
        self.tol = tol
        self.stability = stability
        self.gas_boundary_resolution = gas_boundary_resolution
        self.layer_name = layer_name
        self.layer_num = layer_num

        assert len(self.layer_name) > 0 and layer_num > 0

        log_dist(
            f'enabled eigenvalue with verbose={verbose}, max_iter={max_iter}, tol={tol}, stability={stability}, gas_boundary_resolution={gas_boundary_resolution}, layer_name={layer_name}, layer_num={layer_num}',
            ranks=[0])

    # Replace all nan/pos-inf/neg-inf to zero
    # TODO: Pytorch new version may add this function, replace this one by then.
    def nan_to_num(self, x):
        device = x.device
        x = x.cpu().numpy()
        x = np.nan_to_num(x=x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(x).to(device)

    def normalize(self, v):
        norm_squared = self.inner_product(v, v)
        norm = norm_squared**0.5 + self.stability
        normalized_vectors = [vector / norm for vector in v]
        normalized_vectors = [self.nan_to_num(vector) for vector in normalized_vectors]
        return normalized_vectors

    def inner_product(self, xs, ys):
        return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])

    def get_layers(self, module):
        scope_names = self.layer_name.split('.')
        assert len(scope_names) > 0

        m = module
        for name in scope_names:
            assert hasattr(m, name), "layer_name configuration is invalid."
            m = getattr(m, name)

        return m

    def compute_eigenvalue(self, module, device=None, scale=1.0):
        block_eigenvalue = []
        param_keys = []
        layers = self.get_layers(module)

        for block in range(self.layer_num):
            model_block = layers[block]

            # We found this randn() has obvious accuracy impact in some cases, save/recover random state here.
            rng_state = torch.random.get_rng_state()
            if device is None:
                v = [
                    torch.randn(p.size()) for p in model_block.parameters()
                    if p.grad is not None and p.grad.grad_fn is not None
                ]
            else:
                v = [
                    torch.randn(p.size(), device=device) for p in model_block.parameters()
                    if p.grad is not None and p.grad.grad_fn is not None
                ]
            torch.random.set_rng_state(rng_state)

            grads = [
                param.grad for param in model_block.parameters()
                if param.grad is not None and param.grad.grad_fn is not None
            ]
            params = [
                param for param in model_block.parameters()
                if param.grad is not None and param.grad.grad_fn is not None
            ]

            layer_keys = [id(p) for p in model_block.parameters()]
            param_keys.append(layer_keys)

            v = self.normalize(v)

            # Disable eigenvalue if the model doesn't support second order gradients computation,
            # e.g. when enabling DS transformer kernel.
            if len(grads) == 0 or len(params) == 0:
                log_dist(f'The model does NOT support eigenvalue computation.', ranks=[0], level=logging.WARNING)
                return []

            i = 0
            eigenvalue_current, eigenvalue_previous = 1., 0.

            while (i < self.max_iter) and abs(eigenvalue_current) > 0 and (abs(
                (eigenvalue_current - eigenvalue_previous) / eigenvalue_current) >=
                                                                           self.tol):  # test convergence criteria
                eigenvalue_previous = eigenvalue_current

                Hv = torch.autograd.grad(grads, params, grad_outputs=v, only_inputs=True, retain_graph=True)
                #Hv = [hv.float() for hv in Hv]
                Hv = [self.nan_to_num(hv).float() for hv in Hv]

                eigenvalue_current = self.inner_product(Hv, v).item()

                v = self.normalize(Hv)
                v = [x / scale for x in v]
                i += 1

            eigenvalue_current *= scale
            block_eigenvalue.append(eigenvalue_current)

            if self.verbose:
                log_dist(f'block: {block}, power iteration: {i}, eigenvalue: {eigenvalue_current}', ranks=[0])

        block_eigenvalue = self.post_process(block_eigenvalue)

        if self.verbose:
            log_dist(f'post processed block_eigenvalue: {block_eigenvalue}', ranks=[0])

        # {param_id: (eigenvalue, layer_id)}
        ev_dict = {}
        for i, (layer_keys, value) in enumerate(zip(param_keys, block_eigenvalue)):
            ev_dict.update(dict.fromkeys(layer_keys, (value, i)))

        return ev_dict

    # 1. Map all eigenvalues to [0, 1.0].
    # 2. Some layers can't generate valid eigenvalues on fp16 precision, use 1.0 instead.
    def post_process(self, value_list):
        max_value = abs(max(value_list, key=abs))
        return [abs(v) / max_value if v != 0.0 else 1.0 for v in value_list]
