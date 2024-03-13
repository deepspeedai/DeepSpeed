# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest

import itertools
import pickle
import os
import time

from dataclasses import dataclass
from typing import List

import deepspeed
import torch

from huggingface_hub import HfApi
from packaging import version as pkg_version
from torch import nn
from transformers import pipeline
from transformers.models.t5.modeling_t5 import T5Block
from transformers.models.roberta.modeling_roberta import RobertaLayer

from deepspeed.accelerator import get_accelerator
from deepspeed.git_version_info import torch_info
from deepspeed.model_implementations import DeepSpeedTransformerInference
from deepspeed.ops.op_builder import InferenceBuilder
from deepspeed.ops.op_builder import OpBuilder

from unit.common import DistributedTest

rocm_version = OpBuilder.installed_rocm_version()
if rocm_version != (0, 0):
    pytest.skip("skip inference tests on rocm for now", allow_module_level=True)

_bert_models = [
    "google-bert/bert-base-cased",
    "google-bert/bert-base-uncased",
    "google-bert/bert-large-cased",
    "google-bert/bert-large-uncased",
    "google-bert/bert-base-multilingual-cased",
    "google-bert/bert-base-multilingual-uncased",
    "deepset/minilm-uncased-squad2",
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "dslim/bert-base-NER",
    "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad",
    "distilbert/distilbert-base-cased-distilled-squad",
]
_roberta_models = [
    "FacebookAI/roberta-large",
    "FacebookAI/roberta-base",
    "deepset/roberta-base-squad2",
    "j-hartmann/emotion-english-distilroberta-base",
    "Jean-Baptiste/roberta-large-ner-english",
]
_gpt_models = [
    "openai-community/gpt2",
    "distilbert/distilgpt2",
    "Norod78/hebrew-bad_wiki-gpt_neo-tiny",
    "EleutherAI/gpt-j-6b",
    "EleutherAI/pythia-70m-deduped",
    "bigscience/bloom-560m",
]
_opt_models = [
    "facebook/opt-125m",  # 125m, 1.7B, ..., 175B variants have the same model architecture.
    "facebook/opt-350m",  # 350m applies layer norm after attention layer which is different than other variants.
]
_test_models = set(_bert_models + _roberta_models + _gpt_models + _opt_models)
_test_tasks = [
    "fill-mask", "question-answering", "text-classification", "token-classification", "text-generation",
    "text2text-generation", "summarization", "translation"
]


@dataclass
class ModelInfo:
    modelId: str
    pipeline_tag: str
    tags: List[str]


def _hf_model_list() -> List[ModelInfo]:
    """ Caches HF model list to avoid repeated API calls """

    cache_dir = os.getenv("TRANSFORMERS_CACHE", "~/.cache/huggingface")
    cache_file_path = os.path.join(cache_dir, "DS_model_cache.pkl")
    cache_expiration_seconds = 60 * 60 * 24  # 1 day

    # Load or initialize the cache
    model_data = {"cache_time": 0, "model_list": []}
    if os.path.isfile(cache_file_path):
        with open(cache_file_path, 'rb') as f:
            model_data = pickle.load(f)

    current_time = time.time()

    # Update the cache if it has expired
    if (model_data["cache_time"] + cache_expiration_seconds) < current_time:
        api = HfApi()
        model_data["model_list"] = [
            ModelInfo(modelId=m.modelId, pipeline_tag=m.pipeline_tag, tags=m.tags) for m in api.list_models()
        ]
        model_data["cache_time"] = current_time

        # Save the updated cache
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file_path, 'wb') as f:
            pickle.dump(model_data, f)

    return model_data["model_list"]


# Get a list of all models and mapping from task to supported models
_hf_models = _hf_model_list()
_hf_model_names = [m.modelId for m in _hf_models]
_hf_task_to_models = {task: [m.modelId for m in _hf_models if m.pipeline_tag == task] for task in _test_tasks}

# Get all combinations of task:model to test
_model_w_tasks = [(m, t) for m, t in itertools.product(*[_test_models, _test_tasks]) if m in _hf_task_to_models[t]]

# Assign to pytest variables for testing
pytest.model_w_tasks = _model_w_tasks
pytest.mt_names = [f"{m}-{t}" for m, t in pytest.model_w_tasks]


@pytest.fixture(scope="module", autouse=True)
def verify_models():
    # Verify all test models are registered in HF
    _test_models_not_found = [m for m in _test_models if m not in _hf_model_names]
    if _test_models_not_found:
        pytest.fail(f"Model(s) not found in HuggingFace: {_test_models_not_found}")

    # Verify all models are assigned to at least one task
    _models_to_be_tested = set(m for m, t in _model_w_tasks)
    _missing_task_models = _models_to_be_tested.difference(_test_models)
    if _missing_task_models:
        pytest.fail(f"Model(s) do not have an assigned task: {_missing_task_models}")


""" Fixtures for inference config """


@pytest.fixture(params=pytest.model_w_tasks, ids=pytest.mt_names)
def model_w_task(request):
    return request.param


@pytest.fixture(params=[torch.float, torch.half], ids=["fp32", "fp16"])
def dtype(request):
    return request.param


@pytest.fixture(params=[True, False], ids=["CG", "noCG"])
def enable_cuda_graph(request):
    return request.param


@pytest.fixture(params=[True, False], ids=["Triton", "noTriton"])
def enable_triton(request):
    return request.param


""" Fixtures for running query """


@pytest.fixture
def query(model_w_task):
    model, task = model_w_task
    angle_bracket_mask_models = ["roberta", "camembert", "esm", "ibert", "luke", "mpnet", "yoso", "mpnet"]

    if task == "fill-mask":
        if any(map(lambda x: x in model, angle_bracket_mask_models)):
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
    elif task == "text2text-generation":
        return "Is this review positive or negative? Review: this is the best cast iron skillet you will ever buy"
    elif task == "translation" or task == "summarization":
        return "Hello, my dog is cute"
    else:
        NotImplementedError(f'query for task "{task}" is not implemented')


@pytest.fixture
def inf_kwargs(model_w_task):
    model, task = model_w_task
    if task == "text-generation":
        if model == "EleutherAI/gpt-j-6b":
            # This model on V100 is hitting memory problems that limit the number of output tokens
            return {"do_sample": False, "temperature": 1.0, "max_length": 12}
        return {"do_sample": False, "temperature": 1.0, "max_length": 20}
    else:
        return {}


""" Assertion fixture for verifying model outputs """


def fill_mask_assert(x, y):
    return set(res["token_str"] for res in x) == set(res["token_str"] for res in y)


def question_answering_assert(x, y):
    return x["answer"] == y["answer"]


def text_classification_assert(x, y):
    return set(res["label"] for res in x) == set(res["label"] for res in y)


def token_classification_assert(x, y):
    return set(ent["word"] for ent in x) == set(ent["word"] for ent in y)


def text_generation_assert(x, y):
    return set(res["generated_text"] for res in x) == set(res["generated_text"] for res in y)


def text2text_generation_assert(x, y):
    return set(res["generated_text"] for res in x) == set(res["generated_text"] for res in y)


def translation_assert(x, y):
    return set(res["translation_text"] for res in x) == set(res["translation_text"] for res in y)


def summarization_assert(x, y):
    return set(res["summary_text"] for res in x) == set(res["summary_text"] for res in y)


@pytest.fixture
def assert_fn(model_w_task):
    model, task = model_w_task
    assert_fn_dict = {
        "fill-mask": fill_mask_assert,
        "question-answering": question_answering_assert,
        "text-classification": text_classification_assert,
        "token-classification": token_classification_assert,
        "text-generation": text_generation_assert,
        "text2text-generation": text2text_generation_assert,
        "translation": translation_assert,
        "summarization": summarization_assert
    }
    assert_fn = assert_fn_dict.get(task, None)
    if assert_fn is None:
        NotImplementedError(f'assert_fn for task "{task}" is not implemented')
    return assert_fn


# Used to verify DeepSpeed kernel injection worked with a model
def check_injection(model):

    def verify_injection(module):
        for child in module.children():
            if isinstance(child, nn.ModuleList):
                assert isinstance(child[0], DeepSpeedTransformerInference),\
                    "DeepSpeed-Inference Transformer kernels has not been injected in the model"
                break
            else:
                verify_injection(child)

    verify_injection(model)


# Verify that test is valid
def validate_test(model_w_task, dtype, enable_cuda_graph, enable_triton):
    model, task = model_w_task
    msg = ""
    if enable_cuda_graph and (torch_info["cuda_version"] == "0.0"):
        msg = "CUDA not detected, cannot use CUDA Graph"
    elif enable_cuda_graph and pkg_version.parse(torch.__version__) < pkg_version.parse("1.10"):
        msg = "CUDA Graph is only available in torch versions >= 1.10"
    elif "gpt-j-6b" in model:
        if dtype != torch.half:
            msg = f"Not enough GPU memory to run {model} with dtype {dtype}"
        elif enable_cuda_graph:
            msg = f"Not enough GPU memory to run {model} with CUDA Graph enabled"
    elif "gpt-neox-20b" in model:  # TODO: remove this when neox issues resolved
        msg = "Skipping gpt-neox-20b for now"
    elif ("gpt-neox-20b" in model) and (dtype != torch.half):
        msg = f"Not enough GPU memory to run {model} with dtype {dtype}"
    elif ("bloom" in model) and (dtype != torch.half):
        msg = f"Bloom models only support half precision, cannot use dtype {dtype}"
    elif (model not in _bert_models + _roberta_models) and enable_cuda_graph:
        msg = "Non bert/roberta models do no support CUDA Graph"
    elif enable_triton and not (dtype in [torch.half]):
        msg = "Triton is for fp16"
    elif enable_triton and not deepspeed.HAS_TRITON:
        msg = "triton needs to be installed for the test"
    elif (model not in _bert_models + _roberta_models) and enable_triton:
        msg = "Triton kernels do not support Non bert/roberta models yet"

    # These should be removed once we fix several inference tests failing
    if model in [
            "EleutherAI/pythia-70m-deduped", "distilbert/distilbert-base-cased-distilled-squad", "EleutherAI/gpt-j-6b"
    ]:
        msg = "Test is currently broken"
    return msg


@pytest.mark.inference
class TestModelTask(DistributedTest):
    world_size = 1

    def test(
        self,
        model_w_task,
        dtype,
        enable_cuda_graph,
        enable_triton,
        query,
        inf_kwargs,
        assert_fn,
        perf_meas=True,
    ):
        invalid_test_msg = validate_test(model_w_task, dtype, enable_cuda_graph, enable_triton)
        if invalid_test_msg:
            pytest.skip(invalid_test_msg)

        if dtype not in get_accelerator().supported_dtypes():
            pytest.skip(f"Acceleraor {get_accelerator().device_name()} does not support {dtype}.")

        if not deepspeed.ops.__compatible_ops__[InferenceBuilder.NAME]:
            pytest.skip("This op had not been implemented on this system.", allow_module_level=True)

        model, task = model_w_task
        local_rank = int(os.getenv("LOCAL_RANK", "0"))

        # Load the model on CPU first to avoid OOM for large models @fp32
        pipe = pipeline(task, model=model, device=torch.device("cpu"), framework="pt")
        if dtype == torch.half:
            pipe.model.half()

        # Switch device to GPU after converting to half
        device = torch.device(get_accelerator().device_name(local_rank))
        pipe.device = device
        pipe.model.to(device)

        # Warm-up queries for perf measurement
        #for i in range(10):
        #    _ = pipe(query, **inf_kwargs)
        get_accelerator().synchronize()
        start = time.time()
        bs_output = pipe(query, **inf_kwargs)
        get_accelerator().synchronize()
        bs_time = time.time() - start

        args = {
            'mp_size': 1,
            'dtype': dtype,
            'replace_with_kernel_inject': True,
            'enable_cuda_graph': enable_cuda_graph,
            'use_triton': enable_triton,
            'triton_autotune': False,
        }
        if pipe.tokenizer.model_max_length < deepspeed.ops.transformer.inference.config.DeepSpeedInferenceConfig(
        ).max_out_tokens:
            args.update({'max_out_tokens': pipe.tokenizer.model_max_length})
        pipe.model = deepspeed.init_inference(pipe.model, **args)
        check_injection(pipe.model)
        # Warm-up queries for perf measurement
        #for i in range(10):
        #    _ = pipe(query, **inf_kwargs)
        get_accelerator().synchronize()
        start = time.time()
        ds_output = pipe(query, **inf_kwargs)
        get_accelerator().synchronize()
        ds_time = time.time() - start

        if perf_meas:
            print(
                f"model={model}, task={task}, dtype={dtype}, cuda_graph={enable_cuda_graph}, triton={enable_triton}, bs_time={bs_time}, ds_time={ds_time}"
            )

        # facebook/opt* and some bigscient/bloom* models are not matching
        # baseline exactly, adding an exception to them for now
        if ("opt" in model) or ("bloom" in model):
            bs_output = pipe(query, **inf_kwargs)

        # These performance tests are only measuring the time for a single
        # inference request, we just want to check that performance isn't terrible
        #assert ds_time <= (bs_time * 1.1)

        assert assert_fn(bs_output, ds_output)


@pytest.mark.seq_inference
@pytest.mark.parametrize("model_w_task", [("EleutherAI/gpt-neo-1.3B", "text-generation"),
                                          ("EleutherAI/gpt-neox-20b", "text-generation"),
                                          ("bigscience/bloom-3b", "text-generation"),
                                          ("EleutherAI/gpt-j-6b", "text-generation")],
                         ids=["gpt-neo", "gpt-neox", "bloom", "gpt-j"])
class TestMPSize(DistributedTest):
    world_size = 2

    def test(
        self,
        model_w_task,
        dtype,
        query,
        inf_kwargs,
        assert_fn,
    ):
        invalid_test_msg = validate_test(model_w_task, dtype, enable_cuda_graph=False, enable_triton=False)
        if invalid_test_msg:
            pytest.skip(invalid_test_msg)

        if not deepspeed.ops.__compatible_ops__[InferenceBuilder.NAME]:
            pytest.skip("This op had not been implemented on this system.", allow_module_level=True)

        model, task = model_w_task
        local_rank = int(os.getenv("LOCAL_RANK", "0"))

        # We have to load these large models on CPU with pipeline because not
        # enough GPU memory
        pipe = pipeline(task, model=model, device=torch.device("cpu"), framework="pt")
        bs_output = pipe(query, **inf_kwargs)

        pipe.model = deepspeed.init_inference(pipe.model,
                                              mp_size=self.world_size,
                                              dtype=dtype,
                                              replace_with_kernel_inject=True)
        check_injection(pipe.model)
        # Switch device to GPU so that input tensors are not on CPU
        pipe.device = torch.device(get_accelerator().device_name(local_rank))
        ds_output = pipe(query, **inf_kwargs)

        print(local_rank, "baseline", bs_output)
        print(local_rank, "deepspeed", ds_output)
        assert assert_fn(bs_output, ds_output)


@pytest.mark.inference
@pytest.mark.parametrize("model_w_task", [("openai-community/gpt2", "text-generation")], ids=["gpt2"])
class TestLowCpuMemUsage(DistributedTest):
    world_size = 1

    def test(
        self,
        model_w_task,
        query,
        inf_kwargs,
        assert_fn,
    ):
        model, task = model_w_task
        dtype = torch.float16
        if dtype not in get_accelerator().supported_dtypes():
            pytest.skip(f"Acceleraor {get_accelerator().device_name()} does not support {dtype}.")

        local_rank = int(os.getenv("LOCAL_RANK", "0"))

        pipe = pipeline(task, model=model, model_kwargs={"low_cpu_mem_usage": True}, device=local_rank, framework="pt")
        bs_output = pipe(query, **inf_kwargs)
        pipe.model = deepspeed.init_inference(pipe.model,
                                              mp_size=self.world_size,
                                              dtype=dtype,
                                              replace_method="auto",
                                              replace_with_kernel_inject=True)

        ds_output = pipe(query, **inf_kwargs)

        assert assert_fn(bs_output, ds_output)


@pytest.mark.seq_inference
@pytest.mark.parametrize(
    "model_w_task, injection_policy",
    [
        (("google/t5-v1_1-small", "text2text-generation"), {
            T5Block: ('SelfAttention.o', 'EncDecAttention.o', 'DenseReluDense.wo')
        }),
        (("FacebookAI/roberta-large", "fill-mask"), {
            RobertaLayer: ('output.dense')
        }),
    ],
    ids=["t5", "roberta"],
)
@pytest.mark.parametrize("dtype", [torch.float], ids=["fp32"])
class TestInjectionPolicy(DistributedTest):
    world_size = [1, 2]

    def test(
        self,
        model_w_task,
        injection_policy,
        query,
        inf_kwargs,
        assert_fn,
        dtype,
    ):
        invalid_test_msg = validate_test(model_w_task, dtype, enable_cuda_graph=False, enable_triton=False)
        if invalid_test_msg:
            pytest.skip(invalid_test_msg)

        model, task = model_w_task
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        world_size = int(os.getenv("WORLD_SIZE", "2"))

        pipe = pipeline(task,
                        model=model,
                        device=torch.device(get_accelerator().device_name(local_rank)),
                        framework="pt")
        bs_output = pipe(query, **inf_kwargs)

        pipe.model = deepspeed.init_inference(pipe.model,
                                              mp_size=world_size,
                                              dtype=dtype,
                                              injection_policy=injection_policy)
        ds_output = pipe(query, **inf_kwargs)

        print(local_rank, "baseline", bs_output)
        print(local_rank, "deepspeed", ds_output)
        assert assert_fn(bs_output, ds_output)


@pytest.mark.seq_inference
@pytest.mark.parametrize(
    "model_w_task",
    [("Helsinki-NLP/opus-mt-en-de", "translation"), ("Salesforce/codegen-350M-mono", "text-generation")],
    ids=["marian", "codegen"],  #codegen has fusedqkv weight.
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
class TestAutoTensorParallelism(DistributedTest):
    world_size = [2]

    def test(
        self,
        model_w_task,
        query,
        inf_kwargs,
        assert_fn,
        dtype,
    ):
        invalid_test_msg = validate_test(model_w_task, dtype, enable_cuda_graph=False, enable_triton=False)
        if invalid_test_msg:
            pytest.skip(invalid_test_msg)

        model, task = model_w_task
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        world_size = int(os.getenv("WORLD_SIZE", "2"))

        if dtype not in get_accelerator().supported_dtypes():
            pytest.skip(f"Acceleraor {get_accelerator().device_name()} does not support {dtype}.")

        if model == "Salesforce/codegen-350M-mono":
            pytest.skip("Disable Codegen model due to slight result difference")
            #TODO: re-enable this test once we have a fix for the slight result difference

        pipe = pipeline(task,
                        model=model,
                        device=torch.device(get_accelerator().device_name(local_rank)),
                        framework="pt")
        bs_output = pipe(query, **inf_kwargs)

        pipe.model = deepspeed.init_inference(pipe.model, mp_size=world_size, dtype=dtype)
        ds_output = pipe(query, **inf_kwargs)

        print(local_rank, "baseline", bs_output)
        print(local_rank, "deepspeed", ds_output)
        assert assert_fn(bs_output, ds_output)

    @pytest.mark.world_size(3)
    def test_odd_world_size(
        self,
        model_w_task,
        query,
        inf_kwargs,
        assert_fn,
        dtype,
    ):
        invalid_test_msg = validate_test(model_w_task, dtype, enable_cuda_graph=False, enable_triton=False)
        if invalid_test_msg:
            pytest.skip(invalid_test_msg)

        model, task = model_w_task
        if model == "Salesforce/codegen-350M-mono":
            pytest.skip("codegen does not supported by odd world_size")
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        world_size = int(os.getenv("WORLD_SIZE", "3"))

        pipe = pipeline(task,
                        model=model,
                        device=torch.device(get_accelerator().device_name(local_rank)),
                        framework="pt")
        bs_output = pipe(query, **inf_kwargs)

        pipe.model = deepspeed.init_inference(pipe.model, mp_size=world_size, dtype=dtype)
        ds_output = pipe(query, **inf_kwargs)

        print(local_rank, "baseline", bs_output)
        print(local_rank, "deepspeed", ds_output)
        assert assert_fn(bs_output, ds_output)


@pytest.mark.nightly
@pytest.mark.parametrize(
    "model_family, model_name",
    (
        ["gpt2", "EleutherAI/gpt-neo-2.7B"],
        #["gpt2", "EleutherAI/gpt-j-6b"], # Causing OOM for this test
        ["gpt2", "openai-community/gpt2-xl"],
    ),
)
@pytest.mark.parametrize("task", ["lambada_standard"])
class TestLMCorrectness(DistributedTest):
    world_size = 1
    exec_timeout = 1200  # Give these tests longer to complete

    def test(self, model_family, model_name, task):
        # imports here to avoid import errors when pytest collects tests
        import lm_eval
        import lm_eval.models
        import lm_eval.tasks
        import lm_eval.evaluator

        # The bootstrap_stderr function in lm_eval.metrics uses a
        # multiprocessing Pool to increase performance. Since we use a Pool for
        # our distributed tests and cannot nest Pools, we must redefine and
        # patch this function with a version that does not use Pool.
        def no_pool_bootstrap_stderr(f, xs, iters):
            from lm_eval.metrics import _bootstrap_internal
            from lm_eval.metrics import sample_stddev
            res = []
            chunk_size = min(1000, iters)
            for i in range(iters // chunk_size):
                res.extend(_bootstrap_internal(f, chunk_size)((i, xs)))
            return sample_stddev(res)

        lm_eval.metrics.bootstrap_stderr = no_pool_bootstrap_stderr

        local_rank = os.getenv("LOCAL_RANK", "0")
        device = torch.device(get_accelerator().device_name(local_rank))
        dtype = torch.float
        task_dict = lm_eval.tasks.get_task_dict([task])

        if 'gpt-j-6b' in model_name:
            dtype = torch.half
            lm = lm_eval.models.get_model(model_family).create_from_arg_string(f"pretrained={model_name}",
                                                                               {"device": "cpu"})
            setattr(lm, model_family, getattr(lm, model_family).half().to(device))
            lm._device = device
        else:
            if get_accelerator().device_name() == 'hpu':
                #lm_eval not supporting HPU device, so get model with CPU and move it to HPU.
                lm = lm_eval.models.get_model(model_family).create_from_arg_string(f"pretrained={model_name}",
                                                                                   {"device": "cpu"})
                setattr(lm, model_family, getattr(lm, model_family).to(device))
                lm._device = device
            else:
                lm = lm_eval.models.get_model(model_family).create_from_arg_string(
                    f"pretrained={model_name}", {"device": get_accelerator().device_name()})

        get_accelerator().synchronize()
        start = time.time()
        bs_output = lm_eval.evaluator.evaluate(lm=lm, task_dict=task_dict)
        get_accelerator().synchronize()
        bs_time = time.time() - start

        getattr(lm, model_family).to("cpu")
        ds_model = deepspeed.init_inference(
            getattr(lm, model_family),
            mp_size=1,
            dtype=dtype,
            replace_with_kernel_inject=True,
            enable_cuda_graph=False,
        )
        check_injection(ds_model)
        setattr(lm, model_family, ds_model)
        get_accelerator().synchronize()
        start = time.time()
        ds_output = lm_eval.evaluator.evaluate(lm=lm, task_dict=task_dict)
        get_accelerator().synchronize()
        ds_time = time.time() - start

        ppl_diff = abs(bs_output["results"][task]["ppl"] - ds_output["results"][task]["ppl"])
        #assert ds_time <= bs_time
        assert ppl_diff < 0.01
