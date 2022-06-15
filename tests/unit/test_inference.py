import os
import time
import torch
import pytest
import itertools
import deepspeed
from deepspeed.git_version_info import torch_info
from collections import defaultdict
from .common import distributed_test
from packaging import version as pkg_version
from deepspeed.ops.op_builder import OpBuilder

try:
    import lm_eval
    import lm_eval.models
    import lm_eval.tasks
    from lm_eval.evaluator import evaluate
    from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import HfApi
except ImportError:
    pytest.skip("please install w. [inf] extra to run this test",
                allow_module_level=True)

rocm_version = OpBuilder.installed_rocm_version()
if rocm_version != (0, 0):
    pytest.skip("skip inference tests on rocm for now", allow_module_level=True)

_bert_models = [
    "bert-base-cased",
    "bert-base-uncased",
    "bert-large-cased",
    "bert-large-uncased",
    "bert-base-multilingual-cased",
    "bert-base-multilingual-uncased",
    "deepset/minilm-uncased-squad2",
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "dslim/bert-base-NER",
    "bert-large-uncased-whole-word-masking-finetuned-squad",
    "distilbert-base-cased-distilled-squad",
]
_roberta_models = [
    "roberta-large",
    "roberta-base",
    "deepset/roberta-base-squad2",
    "j-hartmann/emotion-english-distilroberta-base",
    "Jean-Baptiste/roberta-large-ner-english",
]
_gpt_models = [
    "gpt2",
    "distilgpt2",
    "Norod78/hebrew-bad_wiki-gpt_neo-tiny",
    "EleutherAI/gpt-j-6B",
]
_all_models = HfApi().list_models()

test_models = set(_bert_models + _roberta_models + _gpt_models)
test_tasks = [
    "fill-mask",
    "question-answering",
    "text-classification",
    "token-classification",
    "text-generation",
]
pytest.all_models = {
    task: [m.modelId for m in _all_models if m.pipeline_tag == task]
    for task in test_tasks
}

_model_w_tasks = itertools.product(*[test_models, test_tasks])


def _valid_model_task(model_task):
    m, t = model_task
    return m in pytest.all_models[t]


pytest.models_w_tasks = list(filter(_valid_model_task, _model_w_tasks))
pytest.mt_names = [f"{m}-{t}" for m, t in pytest.models_w_tasks]
"""
These fixtures iterate all combinations of tasks and models, dtype, & cuda_graph
"""


@pytest.fixture(params=pytest.models_w_tasks, ids=pytest.mt_names)
def model_w_task(request):
    return request.param


@pytest.fixture(params=[torch.float, torch.half], ids=["fp32", "fp16"])
def dtype(request):
    return request.param


@pytest.fixture(params=[True, False], ids=["CG", "noCG"])
def enable_cuda_graph(request):
    return request.param


"""
This fixture will validate the configuration
"""


@pytest.fixture()
def invalid_model_task_config(model_w_task, dtype, enable_cuda_graph):
    model, task = model_w_task
    if pkg_version.parse(torch.__version__) <= pkg_version.parse("1.2"):
        msg = "DS inference injection doesn't work well on older torch versions"
    elif model not in pytest.all_models[task]:
        msg = f"Not a valid model / task combination: {model} / {task}"
    elif enable_cuda_graph and (torch_info["cuda_version"] == "0.0"):
        msg = "CUDA not detected, cannot use CUDA Graph"
    elif enable_cuda_graph and pkg_version.parse(
            torch.__version__) < pkg_version.parse("1.10"):
        msg = "CUDA Graph is only available in torch versions >= 1.10"
    elif ("gpt-j-6B" in model) and (dtype == torch.float):
        msg = f"Not enough GPU memory to run {model} with dtype {dtype}"
    else:
        msg = ""
    return msg


"""
These fixtures can be used to customize the query, inference args, and assert
statement for each combination of model /task
"""


@pytest.fixture
def query(model_w_task):
    model, task = model_w_task
    if task == "fill-mask":
        if "roberta" in model:
            return "Hello I'm a <mask> model."
        else:
            return "Hell I'm a [MASK] model."
    elif task == "question-answering":
        return {
            "question": "What's my name?",
            "context": "My name is Clara and I live in Berkeley",
        }
    elif task == "text-classification":
        return "DeepSpeed is the greatest"
    elif task == "token-classification":
        return "My name is jean-baptiste and I live in montreal."
    elif task == "text-generation":
        return "DeepSpeed is the greatest"
    else:
        NotImplementedError(f'query for task "{task}" is not implemented')


@pytest.fixture
def inf_kwargs(model_w_task):
    model, task = model_w_task
    if task == "text-generation":
        return {"do_sample": False}
    else:
        return {}


@pytest.fixture
def assert_fn(model_w_task):
    model, task = model_w_task
    if task == "fill-mask":
        return lambda x, y: set(res["token_str"] for res in x) == set(
            res["token_str"] for res in y
        )
    elif task == "question-answering":
        return lambda x, y: x["answer"] == y["answer"]
    elif task == "text-classification":
        return lambda x, y: set(res["label"] for res in x) == set(
            res["label"] for res in y
        )
    elif task == "token-classification":
        return lambda x, y: set(ent["word"] for ent in x) == set(
            ent["word"] for ent in y
        )
    elif task == "text-generation":
        return lambda x, y: set(res["generated_text"] for res in x) == set(
            res["generated_text"] for res in y
        )
    else:
        NotImplementedError(f'assert_fn for task "{task}" is not implemented')


"""
Tests
"""


@pytest.mark.inference
def test_model_task(
    model_w_task,
    dtype,
    enable_cuda_graph,
    query,
    inf_kwargs,
    assert_fn,
    invalid_model_task_config,
):
    if invalid_model_task_config:
        pytest.skip(invalid_model_task_config)

    model, task = model_w_task

    @distributed_test(world_size=[1])
    def _go():
        local_rank = int(os.getenv("LOCAL_RANK", "0"))

        if "gpt-j-6B" in model and dtype == torch.half:
            _model = AutoModelForCausalLM.from_pretrained(model)
            tokenizer = AutoTokenizer.from_pretrained(model)
            _model.half()
            pipe = pipeline(
                task,
                model=_model,
                tokenizer=tokenizer,
                device=local_rank,
                framework="pt",
            )
        else:
            pipe = pipeline(task, model=model, device=local_rank, framework="pt")
            if dtype == torch.half:
                pipe.model.half()

        # Warm-up queries for perf measurement
        for i in range(10):
            _ = pipe(query, **inf_kwargs)
        torch.cuda.synchronize()
        start = time.time()
        bs_output = pipe(query, **inf_kwargs)
        torch.cuda.synchronize()
        bs_time = time.time() - start

        pipe.model = deepspeed.init_inference(
            pipe.model,
            mp_size=1,
            dtype=dtype,
            replace_method="auto",
            replace_with_kernel_inject=True,
            enable_cuda_graph=enable_cuda_graph,
        )
        # Warm-up queries for perf measurement
        for i in range(10):
            _ = pipe(query, **inf_kwargs)
        torch.cuda.synchronize()
        start = time.time()
        ds_output = pipe(query, **inf_kwargs)
        torch.cuda.synchronize()
        ds_time = time.time() - start

        if task == "text-generation":
            bs_output = pipe(query, **inf_kwargs)

        # These performance tests are only measuring the time for a single
        # inference request, we just want to check that performance isn't terrible
        assert ds_time <= (bs_time * 1.1)
        assert assert_fn(bs_output, ds_output)

    _go()


@pytest.mark.nightly
@pytest.mark.parametrize(
    "model_family, model_name",
    (
        ["gpt2",
         "EleutherAI/gpt-neo-2.7B"],
        ["gpt2",
         "EleutherAI/gpt-j-6B"],
        ["gpt2",
         "gpt2-xl"],
    ),
)
@pytest.mark.parametrize("task", ["lambada"])
def test_lm_correctness(model_family, model_name, task):
    @distributed_test(world_size=[1])
    def _go():
        local_rank = os.getenv("LOCAL_RANK", "0")
        device = torch.device(f"cuda:{local_rank}")
        dtype = torch.float
        task_dict = lm_eval.tasks.get_task_dict([task])

        if 'gpt-j-6B' in model_name:
            dtype = torch.half
            lm = lm_eval.models.get_model(model_family).create_from_arg_string(
                f"pretrained={model_name}",
                {"device": "cpu"})
            setattr(lm, model_family, getattr(lm, model_family).half().to(device))
            lm._device = device
        else:
            lm = lm_eval.models.get_model(model_family).create_from_arg_string(
                f"pretrained={model_name}",
                {"device": f"cuda:{local_rank}"})

        torch.cuda.synchronize()
        start = time.time()
        bs_output = evaluate(lm=lm, task_dict=task_dict)
        torch.cuda.synchronize()
        bs_time = time.time() - start

        ds_model = deepspeed.init_inference(
            getattr(lm,
                    model_family),
            mp_size=1,
            dtype=dtype,
            replace_method="auto",
            replace_with_kernel_inject=True,
            enable_cuda_graph=False,
        )
        setattr(lm, model_family, ds_model)
        torch.cuda.synchronize()
        start = time.time()
        ds_output = evaluate(lm=lm, task_dict=task_dict)
        torch.cuda.synchronize()
        ds_time = time.time() - start

        ppl_diff = abs(bs_output["results"][task]["ppl"] -
                       ds_output["results"][task]["ppl"])
        assert ds_time <= bs_time
        assert ppl_diff < 0.01

    _go()
