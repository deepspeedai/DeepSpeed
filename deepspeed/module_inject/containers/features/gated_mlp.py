# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from abc import abstractmethod

from .hybrid_engine import HybridEngineContainer


class HybridGatedMLPContainer(HybridEngineContainer):
    """
    The HybridGatedMLPContainer supports models for which the first MLP layer
    is represented with two separate weights, one for the activation function
    and one for the gating function.
    """

    def set_mlp(self, _h4h_w, _h4h_b, _4hh_w, _4hh_b):
        super().set_mlp(_h4h_w, _h4h_b, _4hh_w, _4hh_b)
        self.set_mlp_gate()

    @abstractmethod
    def set_mlp_gate(self):
        """
        In `set_mlp_gate`, it is necessary to populate the following variables (where appropriate)
        for the given model:
            self.inter_up_w: inter up weight
            self.inter_up_b: inter up bias
            self.inter_gate_w: inter gate weight
            self.inter_gate_b: inter gate bias
        If the parameter does not exist in the original model, set the attribute to None.
        """
        raise NotImplementedError("A set_mlp_gate() function must be defined in the model container \
                                    in order to set the unfused inter up and gate tensors.")

    def mlp_inter_mp(self, mp_replace, reversed_dim=False):
        # Only need to alter behavior if we can't do the normal destructive copy
        if self.module.mlp.inter_w is None:
            params = [
                (self.module.mlp.inter_up_w, self.inter_up_w),
                (self.module.mlp.inter_up_b, self.inter_up_b),
                (self.module.mlp.inter_gate_w, self.inter_gate_w),
                (self.module.mlp.inter_gate_b, self.inter_gate_b),
            ]
            for dst, src in params:
                dst = mp_replace.copy(dst[:self.inter_up_w.shape[0] // mp_replace.mp_size],
                                      src,
                                      int8=reversed_dim,
                                      allocate_tensor=reversed_dim) if src is not None else None
        else:
            self.module.mlp.inter_w = mp_replace.strided_copy(self.module.mlp.inter_w,
                                                              self._h4h_w,
                                                              num_splits=2,
                                                              int8=reversed_dim)
            self.module.mlp.inter_b = mp_replace.strided_copy(self.module.mlp.inter_b,
                                                              self._h4h_b,
                                                              num_splits=2,
                                                              int8=reversed_dim)

    def release_mlp(self):
        super().release_mlp()
        gated_mlp_params = [
            (self.module.mlp.inter_up_w, self.inter_up_w),
            (self.module.mlp.inter_up_b, self.inter_up_b),
            (self.module.mlp.inter_gate_w, self.inter_gate_w),
            (self.module.mlp.inter_gate_b, self.inter_gate_b),
        ]

        self._release_params(gated_mlp_params)

    def reset_mlp(self):
        self._h4h_w.data[:self.inter_up_w.shape[0]] = self.inter_up_w.data
        self._h4h_w.data[self.inter_up_w.shape[0]:] = self.inter_gate_w.data

        if self.inter_up_b is not None:
            self._h4h_b.data[:self.inter_up_b.shape[0]] = self.inter_up_b.data
            self._h4h_b.data[self.inter_up_b.shape[0]:] = self.inter_gate_b.data

        inter_data = [self.inter_up_w.data, self.inter_gate_w.data]
        if self.inter_up_b is not None:
            inter_data.extend([self.inter_up_b.data, self.inter_gate_b.data])

        self.inter_up_w.data = self._h4h_w.data[:self.inter_up_w.shape[0]]
        self.inter_gate_w.data = self._h4h_w.data[self.inter_up_w.shape[0]:]

        if self.inter_up_b is not None:
            self.inter_up_b.data = self._h4h_b.data[:self.inter_up_b.shape[0]]
            self.inter_gate_b.data = self._h4h_b.data[self.inter_up_b.shape[0]:]

        for data in inter_data:
            del data

    def set_mlp_params_wo_copy(self, Z3_enabled=False):
        self.module.mlp.output_w = self._4hh_w
        self.module.mlp.output_b = self._4hh_b

        if not Z3_enabled:
            # In initialize_tensors, we create a fused inter projection with the appropriate shape
            # and copy the up projection and gate projection into it
            self.module.mlp.inter_w = self._h4h_w
            self.module.mlp.inter_b = self._h4h_b

            self.inter_up_w.data = self._h4h_w[:self.inter_up_w.shape[0], :]
            self.inter_gate_w.data = self._h4h_w[self.inter_up_w.shape[0]:, :]

            if self.inter_up_b is not None:
                self.inter_up_b.data = self._h4h_b[:self.inter_up_w.shape[0]] if self._h4h_b is not None else None
                self.inter_gate_b.data = self._h4h_b[self.inter_up_w.shape[0]:] if self._h4h_b is not None else None
        else:
            self.module.mlp.inter_up_w = self.inter_up_w
            self.module.mlp.inter_up_b = self.inter_up_b
            self.module.mlp.inter_gate_w = self.inter_gate_w
            self.module.mlp.inter_gate_b = self.inter_gate_b

    def get_mlp_params(self):
        params = super().get_mlp_params()
        params.extend([self.inter_up_w, self.inter_up_b, self.inter_gate_w, self.inter_gate_b])
        return params
