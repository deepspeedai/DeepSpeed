'''
Copyright 2022 The Microsoft DeepSpeed Team
'''
from ..policy import DSPolicy


class VAEPolicy(DSPolicy):
    def __init__(self):
        super().__init__()
        try:
            import diffusers
            self._orig_layer_class = diffusers.models.vae.AutoencoderKL
        except ImportError:
            self._orig_layer_class = None

    def match(self, module):
        return isinstance(module, self._orig_layer_class)

    def apply(self, module, enable_cuda_graph=True):
        # TODO(cmikeh2): Enable cuda graph should be an inference configuration
        from ...model_implementations.diffusers.vae import DSVAE
        return DSVAE(module, enable_cuda_graph=enable_cuda_graph)

    # NOTE (lekurile): Should we have a diffusers policy class?
    def attention(self):
        pass
