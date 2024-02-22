# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import os
import torch
from deepspeed.accelerator import get_accelerator


@pytest.mark.evaluation
@pytest.mark.parametrize("model_name", ["codellama/CodeLlama-7b-Python-hf"])
def test_human_eval(model_name):
    import mii
    import numpy
    from transformers import pipeline
    from human_eval.data import write_jsonl, read_problems
    from human_eval.evaluation import evaluate_functional_correctness

    def generate_base_completion(pipe, problem_prompt: str) -> str:
        return pipe(problem_prompt, do_sample=True)[0]["generated_text"]

    def generate_mii_completion(pipe, problem_prompt: str) -> str:
        return pipe(problem_prompt, max_new_tokens=512)[0].generated_text

    def generate_samples(pipe, generation_function):
        samples = [
            dict(task_id=task_id, completion=generation_function(pipe, problems[task_id]["prompt"]))
            for task_id in problems for _ in range(num_samples_per_task)
        ]
        return samples

    # Loading Problems
    problems = read_problems("../../human-eval/data/HumanEval.jsonl.gz")
    num_samples_per_task = 20

    # Initializing HuggingFace Pipeline
    local_rank = os.getenv("LOCAL_RANK", "0")
    device = torch.device(get_accelerator().device_name(local_rank))
    base_pipe = pipeline(model=model_name,
                         device=torch.device(get_accelerator().device_name(local_rank)),
                         max_length=512,
                         return_full_text=False)

    # Generating Base Samples
    base_samples = generate_samples(base_pipe, generate_base_completion)

    # Base Pipeline Teardown
    del base_pipe
    get_accelerator().empty_cache()

    # Initializing DeepSpeed-MII Pipeline
    mii_pipe = mii.pipeline(model_name)

    # Generating MII Samples
    mii_samples = generate_samples(mii_pipe, generate_mii_completion)

    # MII Pipeline Teardown
    mii_pipe.destroy()

    # Writing Samples
    write_jsonl("base_samples.jsonl", base_samples)
    write_jsonl("mii_samples.jsonl", mii_samples)

    # Evaluating Samples
    base_results = evaluate_functional_correctness("base_samples.jsonl")
    mii_results = evaluate_functional_correctness("mii_samples.jsonl")

    # Executing Assertions
    for key in base_results.keys():
        assert numpy.allclose(base_results[key], mii_results[key], rtol=0.10), \
            f"Base result: {base_results[key]}, MII result: {mii_results[key]}, outside of rtol."
