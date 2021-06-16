from .builder import CUDAOpBuilder


class InferenceBuilder(CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_TRANSFORMER_INFERENCE"
    NAME = "transformer_inference"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f'deepspeed.ops.transformer_inference.{self.NAME}_op'

    def sources(self):
        return [
            'csrc/transformer/inference/csrc/pt_binding.cpp',
            'csrc/transformer/inference/csrc/gelu.cu',
            'csrc/transformer/inference/csrc/normalize.cu',
            'csrc/transformer/inference/csrc/softmax.cu',
            'csrc/transformer/inference/csrc/dequantize.cu',
        ]

    def include_paths(self):
        return ['csrc/transformer/inference/includes']
