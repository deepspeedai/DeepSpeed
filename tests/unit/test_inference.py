import os
import torch
import pytest
import deepspeed
from collections import defaultdict
from transformers import pipeline
from .common import distributed_test
from packaging import version as pkg_version

pytest.task_query_dict = {
    "fill-mask":
    defaultdict(
        lambda: "Hello I'm a [MASK] model.",
        {"roberta-base": "Hello I'm a <mask> model."},
    ),
    "question-answering":
    defaultdict(lambda: {
        "question": "What is the greatest?",
        "context": "DeepSpeed is the greatest",
    }),
    "text-classification":
    defaultdict(lambda: "DeepSpeed is the greatest"),
    "token-classification":
    defaultdict(lambda: "My name is jean-baptiste and I live in montreal."),
    "text-generation":
    defaultdict(lambda: "DeepSpeed is the greatest"),
}
pytest.task_model_dict = {
    "fill-mask": {
        "bert": "bert-base-cased",
        "roberta": "roberta-base"
    },
    "question-answering": {
        "bert": "deepset/minilm-uncased-squad2",
        "roberta": "deepset/roberta-base-squad2",
    },
    "text-classification": {
        "bert": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "roberta": "j-hartmann/emotion-english-distilroberta-base",
    },
    "token-classification": {
        "bert": "dslim/bert-base-NER",
        "roberta": "Jean-Baptiste/roberta-large-ner-english",
    },
    "text-generation": {
        "gpt2": "distilgpt2",
        "gpt_neo": "Norod78/hebrew-bad_wiki-gpt_neo-tiny",
        "gptj": "EleutherAI/gpt-j-6B",
    },
}


@pytest.fixture
def model(task, model_family):
    if model_family not in pytest.task_model_dict[task]:
        pytest.skip(f"No models in family {model_family} for task {task}")
    return pytest.task_model_dict[task][model_family]


@pytest.fixture
def query(task, model):
    return pytest.task_query_dict[task][model]


@pytest.mark.parametrize(
    "task",
    (
        "fill-mask",
        "question-answering",
        "text-classification",
        "token-classification",
        "text-generation",
    ),
)
@pytest.mark.parametrize("model_family", ("bert", "roberta", "gpt2", "gpt_neo"))
def test_model_task_inject(task, model, query, dtype=torch.float):
    if pkg_version.parse(torch.__version__) <= pkg_version.parse('1.2'):
        pytest.skip("DS inference injection doesn't work well on older torch versions")

    @distributed_test(world_size=[1])
    def _go():
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        generator = pipeline(task, model=model, device=local_rank)

        generator.model = deepspeed.init_inference(
            generator.model,
            mp_size=world_size,
            dtype=dtype,
            replace_method="auto",
            replace_with_kernel_inject=True,
        )

        response = generator(query)

    _go()


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
