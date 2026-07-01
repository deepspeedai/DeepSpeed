# Copyright (c) Snowflake.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import shlex
from pathlib import Path

import modal

ROOT_PATH = Path(__file__).parents[1]
DEFAULT_MODAL_TORCH_PRESET = "2.10.0-cuda12.8"
DEFAULT_MODAL_TRANSFORMERS_SOURCE = "git"
MODAL_TORCH_PRESETS = {
    "2.7.1-cuda12.8": {
        "image": "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel",
        "torch_package": "torch==2.7.1",
        "torchvision_package": "torchvision==0.22.1",
        "torch_test_version": "2.7",
        "cuda_test_version": "12.8",
    },
    "2.8.0-cuda12.8": {
        "image": "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel",
        "torch_package": "torch==2.8.0",
        "torchvision_package": "torchvision==0.23.0",
        "torch_test_version": "2.8",
        "cuda_test_version": "12.8",
    },
    "2.9.1-cuda12.8": {
        "image": "pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel",
        "torch_package": "torch==2.9.1",
        "torchvision_package": "torchvision==0.24.1",
        "torch_test_version": "2.9",
        "cuda_test_version": "12.8",
    },
    "2.10.0-cuda12.8": {
        "image": "pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel",
        "torch_package": "torch==2.10.0",
        "torchvision_package": "torchvision==0.25.0",
        "torch_test_version": "2.10",
        "cuda_test_version": "12.8",
    },
    "2.11.0-cuda12.8": {
        "image": "pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel",
        "torch_package": "torch==2.11.0",
        "torchvision_package": "torchvision==0.26.0",
        "torch_test_version": "2.11",
        "cuda_test_version": "12.8",
    },
}
PYTORCH_CUDA_128_INDEX_URL = "https://download.pytorch.org/whl/cu128"


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


def resolve_modal_transformers_config():
    transformers_source = os.environ.get("MODAL_TRANSFORMERS_SOURCE") or DEFAULT_MODAL_TRANSFORMERS_SOURCE
    supported_sources = {"requirements", "git"}
    if transformers_source not in supported_sources:
        supported = ", ".join(sorted(supported_sources))
        raise ValueError(
            f"Unsupported MODAL_TRANSFORMERS_SOURCE={transformers_source!r}; supported values: {supported}")

    transformers_ref = os.environ.get("MODAL_TRANSFORMERS_REF") or ""
    if transformers_source == "git" and not transformers_ref:
        transformers_ref = "main"

    return {
        "source": transformers_source,
        "ref": transformers_ref,
    }


def transformers_override_commands():
    if MODAL_TRANSFORMERS_CONFIG["source"] == "requirements":
        return ()

    transformers_ref = shlex.quote(MODAL_TRANSFORMERS_CONFIG["ref"])
    return (
        "rm -rf /tmp/transformers",
        "git clone --filter=blob:none https://github.com/huggingface/transformers /tmp/transformers",
        "cd /tmp/transformers && "
        f"git checkout {transformers_ref} && "
        "resolved_ref=$(git rev-parse HEAD) && "
        "echo \"Resolved Transformers git ref: ${resolved_ref}\" && "
        "pip install .",
    )


def torch_package_reinstall_command():
    command = [
        "pip",
        "install",
        "--force-reinstall",
        "--no-cache-dir",
        "--index-url",
        PYTORCH_CUDA_128_INDEX_URL,
        MODAL_TORCH_CONFIG["torch_package"],
        MODAL_TORCH_CONFIG["torchvision_package"],
    ]
    return " ".join(shlex.quote(part) for part in command)


MODAL_TORCH_CONFIG = resolve_modal_torch_config()
MODAL_TRANSFORMERS_CONFIG = resolve_modal_transformers_config()
MODAL_TORCH_IMAGE = MODAL_TORCH_CONFIG["image"]
MODAL_TORCH_TEST_VERSION = MODAL_TORCH_CONFIG["torch_test_version"]
MODAL_CUDA_TEST_VERSION = MODAL_TORCH_CONFIG["cuda_test_version"]

# Full test sub-tree this workflow runs by default.
DEFAULT_TEST_TARGET = "tests/unit/v1/"


def resolve_selected_tests():
    """pytest targets selected by the diff-driven fetcher (ci/tests_fetcher.py).

    The fetcher writes one target per line (individual test files for a narrowed
    run, or just ``tests/unit/v1`` for a full run). The ``collect-tests`` CI job
    produces this file and the ``deploy`` job downloads it into place before
    ``modal run``. We read it here on the runner (at image-build time) and bake the
    targets into the image env so the remote ``pytest`` picks them up.

    Falls back to the whole v1 suite when the file is missing or empty (local runs,
    or selection skipped), so behavior is unchanged unless a selection is provided.
    """
    env_path = os.environ.get("DS_TEST_LIST_FILE", "")
    list_file = Path(env_path) if env_path else ROOT_PATH / "ci" / ".test_selection" / "test_list.txt"
    if not list_file.is_absolute():
        list_file = ROOT_PATH / list_file
    try:
        targets = [line.strip() for line in list_file.read_text().splitlines() if line.strip()]
    except OSError:
        targets = []
    return targets or [DEFAULT_TEST_TARGET]


SELECTED_TESTS = resolve_selected_tests()

# yapf: disable
image = (modal.Image
         .from_registry(MODAL_TORCH_IMAGE, add_python="3.10")
         .env({
             "MODAL_TORCH_PRESET": MODAL_TORCH_CONFIG["preset"],
             "MODAL_TRANSFORMERS_SOURCE": MODAL_TRANSFORMERS_CONFIG["source"],
             "MODAL_TRANSFORMERS_REF": MODAL_TRANSFORMERS_CONFIG["ref"],
             "DS_SELECTED_TESTS": " ".join(SELECTED_TESTS),
         })
         .run_commands("apt update && apt install -y git libaio-dev")
         .pip_install_from_requirements(ROOT_PATH / "requirements/requirements.txt", gpu="any")
         .pip_install_from_requirements(ROOT_PATH / "requirements/requirements-dev.txt", gpu="any")
         .pip_install_from_requirements(ROOT_PATH / "requirements/requirements-deepcompile.txt", gpu="any")
         .run_commands(torch_package_reinstall_command())
        )

transformers_commands = transformers_override_commands()
if transformers_commands:
    image = image.run_commands(*transformers_commands)

image = (image
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
        [
            "python",
            "-c",
            "import json, torch, torchvision, transformers; "
            "print('Modal Python package versions: ' + json.dumps({"
            "'torch': torch.__version__, "
            "'torch_cuda': torch.version.cuda, "
            "'torchvision': torchvision.__version__, "
            "'transformers': transformers.__version__"
            "}, sort_keys=True))",
        ],
        check=True,
        cwd=ROOT_PATH / ".",
    )
    selected_tests = os.environ.get("DS_SELECTED_TESTS", DEFAULT_TEST_TARGET).split()
    print(f"Running pytest on diff-selected targets: {selected_tests}")
    subprocess.run(
        [
            "pytest",
            "-n",
            "4",
            "--verbose",
            *selected_tests,
            f"--torch_ver={MODAL_TORCH_TEST_VERSION}",
            f"--cuda_ver={MODAL_CUDA_TEST_VERSION}",
        ],
        check=True,
        cwd=ROOT_PATH / ".",
    )
