import os
import torch
import pytest
import deepspeed
from transformers import pipeline
from .common import distributed_test
from packaging import version as pkg_version


@pytest.mark.parametrize("dtype", [(torch.float), (torch.half)])
def test_gpt2_inject(dtype):
    if pkg_version.parse(torch.__version__) <= pkg_version.parse('1.2'):
        pytest.skip("DS inference injection doesn't work well on older torch versions")

    @distributed_test(world_size=[1])
    def _go():
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        generator = pipeline("text-generation", model="gpt2", device=local_rank)

        generator.model = deepspeed.init_inference(
            generator.model,
            mp_size=world_size,
            dtype=dtype,
            replace_method="auto",
            replace_with_kernel_inject=True,
        )

        prompt = "DeepSpeed is"
        string_1 = generator(prompt, do_sample=False, max_length=128)
        string_2 = generator(prompt, do_sample=False, max_length=128)
        assert string_1 == string_2

    _go()
