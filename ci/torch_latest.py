# Copyright (c) Snowflake.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from pathlib import Path

import modal

ROOT_PATH = Path(__file__).parents[1]

# yapf: disable
image = (modal.Image
         .from_registry("pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel", add_python="3.10")
         .run_commands("apt update && apt install -y libaio-dev")
         .pip_install_from_requirements(ROOT_PATH / "requirements/requirements.txt", gpu="any")
         .pip_install_from_requirements(ROOT_PATH / "requirements/requirements-dev.txt", gpu="any")
         .pip_install_from_requirements(ROOT_PATH / "requirements/requirements-deepcompile.txt", gpu="any")
         .add_local_dir(ROOT_PATH , remote_path="/root/", copy=True)
         .run_commands("pip install /root")
         .add_local_dir(ROOT_PATH / "accelerator", remote_path="/root/deepspeed/accelerator")
         .add_local_dir(ROOT_PATH / "csrc", remote_path="/root/deepspeed/ops/csrc")
         .add_local_dir(ROOT_PATH / "op_builder", remote_path="/root/deepspeed/ops/op_builder")
        )


app = modal.App("deepspeedai-torch-latest-ci", image=image)


@app.function(
    gpu="l40s:2",
    timeout=1800,
)
def pytest():
    import subprocess
    subprocess.run(
        "pytest -n 4 --verbose tests/unit/v1/ --torch_ver=2.9 --cuda_ver=12.8".split(),
        check=True,
        cwd=ROOT_PATH / ".",
    )
