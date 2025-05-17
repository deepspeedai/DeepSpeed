from pathlib import Path

import modal

ROOT_PATH = Path(__file__).parent.parent

image = (modal.Image.from_registry(
    "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel",
    add_python="3.10")
    .run_commands("apt update && apt install -y libaio-dev")
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


app = modal.App("deepspeedai-ci", image=image)


@app.function(
    gpu="a10g:2",
    # secrets=[modal.Secret.from_local_environ(["HF_TOKEN"])],
    timeout=1500,
)
def pytest():
    import subprocess
    subprocess.run(
        "pytest --disable-warnings --instafail -n 4 tests".split(),
        check=True,
        cwd=ROOT_PATH / ".",
    )
