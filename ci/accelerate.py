# Copyright (c) Snowflake.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
from pathlib import Path

import modal

ROOT_PATH = Path(__file__).parents[1]

# Tree under test. On merge queue runs this launcher is checked out from
# trusted master, so the merged candidate is fetched separately and pointed at
# here. Remotely the env var is unset and this module is materialized inside
# the image at /root, where it must keep resolving to ROOT_PATH.
CANDIDATE_PATH = Path(os.environ.get("DS_CI_CANDIDATE_ROOT") or ROOT_PATH)

# yapf: disable
image = (modal.Image
         .from_registry("pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel", add_python="3.10")
         .apt_install("git")
         .pip_install("uv")
         # uv_pip_install already includes --compile-bytecode
         .uv_pip_install("datasets==3.6.0", extra_options="--system")
         .pip_install_from_requirements(CANDIDATE_PATH / "requirements/requirements.txt", gpu="any")
         .pip_install_from_requirements(CANDIDATE_PATH / "requirements/requirements-dev.txt", gpu="any")
         .add_local_dir(CANDIDATE_PATH, remote_path="/root/", copy=True)
         .run_commands("pip install /root")
         .add_local_dir(CANDIDATE_PATH / "accelerator", remote_path="/root/deepspeed/accelerator")
         .add_local_dir(CANDIDATE_PATH / "csrc", remote_path="/root/deepspeed/ops/csrc")
         .add_local_dir(CANDIDATE_PATH / "op_builder", remote_path="/root/deepspeed/ops/op_builder")
        )

app = modal.App("deepspeedai-accelerate-ci", image=image)

@app.function(
    gpu="l40s:1",
    timeout=1800,
)
def pytest():
    import subprocess

    cmd = "git clone https://github.com/huggingface/accelerate"
    print(f"running: {cmd}")
    subprocess.run(
        cmd.split(),
        check=True,
        cwd=ROOT_PATH / ".",
    )
    cmd = "uv pip install --system --compile-bytecode ./accelerate[testing]"
    print(f"running: {cmd}")
    subprocess.run(
        cmd.split(),
        check=True,
        cwd=ROOT_PATH / ".",
    )

    cmd = "pytest ./accelerate/tests/deepspeed"
    print(f"running: {cmd}")
    subprocess.run(
        cmd.split(),
        check=True,
        cwd=ROOT_PATH / ".",
    )
