# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest


@pytest.mark.inference
class TestInferenceImport:
    """CPU-only tests for the public inference package import surface."""

    def test_transformer_inference_is_lazy_imported(self):
        # Regression test for https://github.com/deepspeedai/DeepSpeed/issues/7159
        # Importing the inference package must not trigger a circular import even
        # when Triton is installed, and the legacy public symbol must stay reachable.
        from deepspeed.model_implementations.transformers.ds_transformer import (
            DeepSpeedTransformerInference as DirectTransformerInference, )
        from deepspeed.ops.transformer.inference import (
            DeepSpeedTransformerInference as OpsTransformerInference, )

        assert OpsTransformerInference is DirectTransformerInference
