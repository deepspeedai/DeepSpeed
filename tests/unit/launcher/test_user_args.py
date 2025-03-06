# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import subprocess

from deepspeed.accelerator import get_accelerator

if not get_accelerator().is_available():
    pytest.skip("only supported in accelerator environments.", allow_module_level=True)

user_arg_test_script = """import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--world_size", type=int, default=1)
args = parser.parse_args()
print("ARG PARSE SUCCESS")
"""


@pytest.fixture(scope="function")
def user_script_fp(tmpdir):
    script_fp = tmpdir.join("user_arg_test.py")
    with open(script_fp, "w") as f:
        f.write(user_arg_test_script)
    return script_fp


@pytest.fixture(scope="function")
def cmd(user_script_fp, prompt, multi_node):
    if multi_node:
        cmd = ("deepspeed", "--force_multi", "--num_nodes", "1", "--num_gpus", "1", user_script_fp, "--prompt", prompt)
    else:
        cmd = ("deepspeed", "--num_nodes", "1", "--num_gpus", "1", user_script_fp, "--prompt", prompt)
    return cmd


@pytest.mark.parametrize("prompt", [
    '''"I am 6' tall"''', """'I am 72" tall'""", """'"translate English to Romanian: "'""",
    '''I'm going to tell them "DeepSpeed is the best"'''
])
@pytest.mark.parametrize("multi_node", [True, False])
def test_user_args(cmd, multi_node):
    if multi_node and get_accelerator().device_name() == "cpu":
        pytest.skip("CPU accelerator does not support this test yet")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    assert "ARG PARSE SUCCESS" in out.decode("utf-8"), f"User args not parsed correctly: {err.decode('utf-8')}"


def test_bash_string_args(tmpdir, user_script_fp):
    bash_script = f"""
    ARGS="--prompt 'DeepSpeed is the best'"
    echo ${{ARGS}}|xargs deepspeed --num_nodes 1 --num_gpus 1 {user_script_fp}
    """

    bash_fp = tmpdir.join("bash_script.sh")
    with open(bash_fp, "w") as f:
        f.write(bash_script)

    p = subprocess.Popen(["bash", bash_fp], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    assert "ARG PARSE SUCCESS" in out.decode("utf-8"), f"User args not parsed correctly: {err.decode('utf-8')}"
