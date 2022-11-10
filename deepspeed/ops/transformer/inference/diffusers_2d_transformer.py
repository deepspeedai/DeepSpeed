'''
Copyright 2022 The Microsoft DeepSpeed Team
'''


class Diffusers2DTransformerConfig():
    def __init__(self, int8_quantization=False):
        self.int8_quantization = int8_quantization
