import torch
import deepspeed
from ..config import DeepSpeedInferenceConfig


class BaseOp(torch.nn.Module):
    inference_cuda_module = None

    def __init__(self, config: DeepSpeedInferenceConfig):
        super(BaseOp, self).__init__()
        self.config = config
        if BaseOp.inference_cuda_module is None:
            builder = deepspeed.ops.op_builder.InferenceBuilder()
            BaseOp.inference_cuda_module = builder.load()
