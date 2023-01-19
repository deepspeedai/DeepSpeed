from abc import ABC


class MetaTensorContainer(ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_meta = False

    def initialize_tensors(self):
        super().initialize_tensors()
        self.is_meta = self.qkvw.is_meta

    def apply_tensor_parallelism(self, mp_replace):
        if self.is_meta:
            if self.qkvb is None:
                self.module.attention.attn_qkvb = None
            if self.dense_b is None:
                self.module.attention.attn_ob = None
        else:
            super().apply_tensor_parallelism(mp_replace)

    def copy_data_to_new_module(self):
        if self.is_meta:
            if self.attn_nw is None:
                self.module.mlp.attn_nw = self.attn_nw
                self.module.mlp.attn_nb = self.attn_nb
        else:
            super().copy_data_to_new_module()

    def transpose(self):
        if not self.is_meta:
            super().transpose()
