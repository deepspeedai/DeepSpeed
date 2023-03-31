# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import subprocess as sp
import os
from math import isclose
import sys
import pytest
import json

sys.path.append("../../../DeepSpeedExamples/BingBertSquad")
import evaluate as eval

squad_dir = "/data/BingBertSquad"
base_dir = "../../../DeepSpeedExamples/BingBertSquad"

script_file_name = "run_squad_deepspeed.sh"
model_file_name = "training_state_checkpoint_162.tar"
eval_file_name = "dev-v1.1.json"
pred_file_name = "predictions.json"

num_gpus = "4"
timeout_sec = 5 * 60 * 60  # 5 hours

eval_version = "1.1"


def create_config_file(tmpdir, zeroenabled=False):
    config_dict = {
        "train_batch_size": 24,
        "train_micro_batch_size_per_gpu": 6,
        "steps_per_print": 10,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 3e-5,
                "weight_decay": 0.0,
                "bias_correction": False
            }
        },
        "gradient_clipping": 1.0,
        "fp16": {
            "enabled": True
        }
    }
    config_dict["zero_optimization"] = zeroenabled

    config_path = os.path.join(tmpdir, 'temp_config.json')
    with open(config_path, 'w') as fd:
        json.dump(config_dict, fd)
    return config_path


def test_e2e_squad_deepspeed_base(tmpdir):
    config_file = create_config_file(tmpdir)

    # base run results => {"exact_match": 83.9829706717124, "f1": 90.71138132004097}
    expected_exact_match = 83.98
    expected_f1 = 90.71

    model_file = os.path.join(squad_dir, model_file_name)
    eval_file = os.path.join(squad_dir, eval_file_name)

    output_dir = os.path.join(tmpdir, "output")
    pred_file = os.path.join(output_dir, pred_file_name)

    proc = sp.Popen(["bash", script_file_name, num_gpus, model_file, squad_dir, output_dir, config_file], cwd=base_dir)

    try:
        proc.communicate(timeout=timeout_sec)

        if os.path.exists(pred_file):
            eval_result = eval.evaluate(eval_version, eval_file, pred_file)

            print("evaluation result: ", json.dumps(eval_result))

            assert isclose(eval_result["exact_match"], expected_exact_match, abs_tol=1e-2)
            assert isclose(eval_result["f1"], expected_f1, abs_tol=1e-2)

        else:
            pytest.fail("Error: Run Failed")

    except sp.TimeoutExpired:
        proc.kill()
        pytest.fail("Error: Timeout")
    except sp.CalledProcessError:
        pytest.fail("Error: Run Failed")


def test_e2e_squad_deepspeed_zero(tmpdir):
    config_file = create_config_file(tmpdir, True)

    # base run results => {"exact_match": 84.1438032166509, "f1": 90.89776136505441}
    expected_exact_match = 84.14
    expected_f1 = 90.89

    model_file = os.path.join(squad_dir, model_file_name)
    eval_file = os.path.join(squad_dir, eval_file_name)

    output_dir = os.path.join(tmpdir, "output")
    pred_file = os.path.join(output_dir, pred_file_name)

    proc = sp.Popen(["bash", script_file_name, num_gpus, model_file, squad_dir, output_dir, config_file], cwd=base_dir)

    try:
        proc.communicate(timeout=timeout_sec)

        if os.path.exists(pred_file):
            eval_result = eval.evaluate(eval_version, eval_file, pred_file)

            print("evaluation result: ", json.dumps(eval_result))

            assert isclose(eval_result["exact_match"], expected_exact_match, abs_tol=1e-2)
            assert isclose(eval_result["f1"], expected_f1, abs_tol=1e-2)

        else:
            pytest.fail("Error: Run Failed")

    except sp.TimeoutExpired:
        proc.kill()
        pytest.fail("Error: Timeout")
    except sp.CalledProcessError:
        pytest.fail("Error: Run Failed")
