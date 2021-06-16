from .builder import CUDAOpBuilder


class QuantizerBuilder(CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_QUANTIZER"
    NAME = "quantizer"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f'deepspeed.ops.quantizer.{self.NAME}_op'

    def sources(self):
        return [
            'csrc/quantization/pt_binding.cpp',
            'csrc/quantization/quantizer.cu',
        ]

    def include_paths(self):
        return ['csrc/includes']
