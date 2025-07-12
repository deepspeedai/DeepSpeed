# Copyright (c) Snowflake.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from pathlib import Path

import modal

ROOT_PATH = Path(__file__).parent.parent.parent

# yapf: disable
image = (modal.Image
         .from_registry("pytorch/pytorch:2.7.0-cuda12.6-cudnn9-devel", add_python="3.10")
         .run_commands("apt update && apt install -y libaio-dev")
        .apt_install("git")
         .run_commands("uv pip install --system --compile-bytecode datasets==3.6.0")
         .run_commands(
                "git clone https://github.com/huggingface/accelerate && \
                uv pip install --system --compile-bytecode accelerate[testing]"
            )
         .run_commands("uv pip install --system --compile-bytecode protobuf")
         .run_commands("pip list")
         .pip_install_from_requirements(ROOT_PATH / "requirements/requirements.txt", gpu="any")
         .pip_install_from_requirements(ROOT_PATH / "requirements/requirements-dev.txt", gpu="any")
         .add_local_dir(ROOT_PATH / "accelerator", remote_path="/root/accelerator")
         .add_local_dir(ROOT_PATH / "accelerator", remote_path="/root/deepspeed/accelerator")
         .add_local_dir(ROOT_PATH / "csrc", remote_path="/root/csrc")
         .add_local_dir(ROOT_PATH / "csrc", remote_path="/root/deepspeed/ops/csrc")
         .add_local_dir(ROOT_PATH / "op_builder", remote_path="/root/op_builder")
         .add_local_dir(ROOT_PATH / "op_builder", remote_path="/root/deepspeed/ops/op_builder")
         .add_local_dir(ROOT_PATH / "tests", remote_path="/root/tests")
        )

app = modal.App("deepspeedai-accelerate-ci", image=image)

@app.function(
    gpu="l40s:4",
    # gpu="a10g:2",
    # secrets=[modal.Secret.from_local_environ(["HF_TOKEN"])],
    timeout=1800,
)
def pytest():
    import subprocess
    subprocess.run(
        "pytest accelerate/tests/deepspeed".split(),
        check=True,
        cwd=ROOT_PATH / ".",
    )
