# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
from functools import reduce

import torch
from torch.fx import Graph, Node

from .fx import get_output_node
from .util import get_param_nodes, get_input_nodes


@dataclass
class DSGraphParam:
    name: str
    shape: torch.Size
    dtype: torch.dtype
    device: torch.device
    node: Node
    allgather_node: Node
    release_node: Node
    param: torch.Tensor
    numel: int = field(init=False)

    def __post_init__(self):
        self.numel = reduce(lambda x, y: x * y, self.shape)


class DSGraphParamManager:

    def __init__(self, fw_graph: Graph, sample_inputs: Any, index_to_ds_ids: List[Tuple[int, int, int]]):
        self._fw_graph = fw_graph
        self._bw_graph = None
        self._params: Dict[str, DSGraphParam] = {}
        self._param_name_to_grad: Dict[str, Node] = {}
        self._ds_ids: Dict[str, int] = {}

        param_nodes = get_param_nodes(fw_graph, index_to_ds_ids)
        self._param_names = [pn.name for pn in param_nodes]
        self._param_indices = [i for i, _, _ in index_to_ds_ids]

        param_inputs = [sample_inputs[i] for i, _, _ in index_to_ds_ids]
        ds_ids = [ds_id for _, ds_id, _ in index_to_ds_ids]
        ds_shapes = [ds_shape for _, _, ds_shape in index_to_ds_ids]

        for pn, pi, ds_id, ds_shape in zip(param_nodes, param_inputs, ds_ids, ds_shapes):
            self._params[pn.name] = DSGraphParam(name=pn.name,
                                                 shape=ds_shape,
                                                 dtype=pi.dtype,
                                                 device=pi.device,
                                                 node=pn,
                                                 allgather_node=None,
                                                 release_node=None,
                                                 param=pi)
            self._ds_ids[pn.name] = ds_id

    def get_bwd_mapping(self, bw_graph: Graph):
        self._bw_graph = bw_graph

        output_node = get_output_node(bw_graph)
        param_nodes_bw = [n for n in self._bw_graph.nodes if n.name in self.param_names]
        grad_outputs = [output_node.args[0][i] for i in self._param_indices]
        param_name_to_grad = {param_name: grad for param_name, grad in zip(self.param_names, grad_outputs)}
        return param_nodes_bw, param_name_to_grad

    @property
    def param_names(self) -> List[str]:
        return self._param_names

    @property
    def params(self) -> Dict[str, DSGraphParam]:
        return self._params

    @property
    def ds_ids(self) -> Dict[str, int]:
        return self._ds_ids

    def get_grad_name(self, param_name) -> str:
        assert self._param_name_to_grad is not None, "Backward graph is not added yet"
        return self._param_name_to_grad[param_name]

    def replace_fake_tensors_with_real_params(self, sample_inputs: List[Any], bw_graph: Graph) -> List[Any]:
        """Replace fake tensors in sample_inputs with real parameters from DSGraphParamManager

        Args:
            sample_inputs: The input tensors that may contain fake tensors
            bw_graph: The backward graph to get parameter mapping from (if in backward pass)
        """
        replaced_inputs = list(sample_inputs)

        # For backward pass, get the parameter nodes and their mapping
        param_nodes_bw, _ = self.get_bwd_mapping(bw_graph)
        param_names_bw = [n.name for n in param_nodes_bw]

        for i, inp in enumerate(get_input_nodes(bw_graph)):
            if inp.name in param_names_bw:
                replaced_inputs[i] = self._params[inp.name].param

        return replaced_inputs
