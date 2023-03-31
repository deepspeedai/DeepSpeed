# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team


class Diffusers2DTransformerConfig():

    def __init__(self, int8_quantization=False):
        self.int8_quantization = int8_quantization
