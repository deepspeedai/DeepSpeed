# Copyright (c) Snowflake.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
from pathlib import Path

import modal

ROOT_PATH = Path(__file__).parents[1]
DEFAULT_MODAL_TORCH_PRESET = "2.9.1-cuda12.8"
MODAL_TORCH_PRESETS = {
    "2.7.1-cuda12.8": {
        "image": "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel",
        "torch_test_version": "2.7",
        "cuda_test_version": "12.8",
    },
    "2.8.0-cuda12.8": {
        "image": "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel",
        "torch_test_version": "2.8",
        "cuda_test_version": "12.8",
    },
    "2.9.1-cuda12.8": {
        "image": "pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel",
        "torch_test_version": "2.9",
        "cuda_test_version": "12.8",
    },
    "2.10.0-cuda12.8": {
        "image": "pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel",
        "torch_test_version": "2.10",
        "cuda_test_version": "12.8",
    },
    "2.11.0-cuda12.8": {
        "image": "pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel",
        "torch_test_version": "2.11",
        "cuda_test_version": "12.8",
    },
}


def resolve_modal_torch_config():
    selected_preset = os.environ.get("MODAL_TORCH_PRESET") or DEFAULT_MODAL_TORCH_PRESET
    try:
        preset_config = MODAL_TORCH_PRESETS[selected_preset]
    except KeyError as exc:
        supported = ", ".join(sorted(MODAL_TORCH_PRESETS))
        raise ValueError(f"Unsupported MODAL_TORCH_PRESET={selected_preset!r}; supported values: {supported}") from exc

    return {
        "preset": selected_preset,
        **preset_config,
    }


MODAL_TORCH_CONFIG = resolve_modal_torch_config()
MODAL_TORCH_IMAGE = MODAL_TORCH_CONFIG["image"]
MODAL_TORCH_TEST_VERSION = MODAL_TORCH_CONFIG["torch_test_version"]
MODAL_CUDA_TEST_VERSION = MODAL_TORCH_CONFIG["cuda_test_version"]

# yapf: disable
image = (modal.Image
         .from_registry(MODAL_TORCH_IMAGE, add_python="3.10")
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
    timeout=3600,
)
def pytest():
    import subprocess
    subprocess.run(
        f"pytest -n 4 --verbose tests/unit/v1/ --torch_ver={MODAL_TORCH_TEST_VERSION} "
        f"--cuda_ver={MODAL_CUDA_TEST_VERSION}".split(),
        check=True,
        cwd=ROOT_PATH / ".",
    )
